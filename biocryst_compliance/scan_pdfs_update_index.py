#!/usr/bin/env python3
"""
scan_pdfs_update_index.py
─────────────────────────
• Extracts DOI from every PDF in 02_sources/pdf/
• Queries CrossRef for metadata (title, journal, authors, year)
• Adds / updates rows in 00_reference_index.csv
• Auto‑generates a 100–120‑word LLM_Summary via GPT‑4o if missing.

Dependencies:
  pip install requests PyPDF2 pandas

Adds two new columns: Abstract (from CrossRef if available) and an empty LLM_Summary
placeholder that can later be filled with a 100–120‑word AI synopsis.
"""

import re, csv, json, requests, pathlib
import os, textwrap, openai
from PyPDF2 import PdfReader
import pandas as pd

PDF_DIR   = pathlib.Path("02_sources/pdf")
INDEX_CSV = pathlib.Path("00_reference_index.csv")
# desired length for the AI summary (≈100–120 words gives clear context without verbosity)
SUMMARY_WORD_LIMIT = 120

# match DOI but stop at whitespace, punctuation or end‑of‑string
doi_re = re.compile(
    r"(10\.\d{4,9}/[-._;()/:A-Z0-9]+?)(?=[\s\"'<>),.;]|$)",
    re.I,
)

def generate_summary(text: str, limit: int = SUMMARY_WORD_LIMIT) -> str:
    """Return an ≈limit‑word plain‑language summary via OpenAI GPT‑4o."""
    prompt = textwrap.dedent(f"""
        Summarise the following article excerpt in about {limit} words.
        Keep medical terms but avoid jargon or filler.

        EXCERPT:
        \"\"\"{text[:4000]}\"\"\"
    """)
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[WARN] LLM summary failed: {e.__class__.__name__}")
        return ""

def extract_doi(pdf_path: pathlib.Path) -> str | None:
    try:
        reader = PdfReader(str(pdf_path))
        first_pages = "".join(page.extract_text() or "" for page in reader.pages[:3])
        match = doi_re.search(first_pages)
        if not match:
            return None
        raw = match.group(0)
        clean = raw

        # 1) remove trailing parentheses, punctuation, whitespace
        clean = re.sub(r"[)\]\}>,.;:\"'\s]+$", "", clean)

        # 2) remove alphabetic tails such as 'J', 'REVIEWS', 'Received'
        #    (1–15 letters at end of string)
        clean = re.sub(r"[A-Za-z]{1,15}$", "", clean)

        # 3) drop any final dot
        clean = clean.rstrip(".")

        return clean
    except Exception as e:
        print(f"[WARN] {pdf_path.name}: {e.__class__.__name__}")
        return None

def crossref_meta(doi: str) -> dict | None:
    url = f"https://api.crossref.org/works/{doi}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            print(f"[WARN] {doi} → HTTP {r.status_code}")
            return None
        item = r.json()["message"]
        first_author = item["author"][0]["family"]
        year = item["issued"]["date-parts"][0][0]
        journal = item["container-title"][0] if item["container-title"] else ""
        title   = item["title"][0][:80]
        short_name = f"{first_author}_{year}"
        abstract = item.get("abstract", "")
        if abstract:
            # strip HTML tags that CrossRef sometimes returns
            abstract = re.sub(r"<[^>]+>", "", abstract).replace("\n", " ").strip()
        return dict(
            ShortName=short_name,
            Year=year,
            FirstAuthor=first_author,
            Journal=journal,
            DOI=doi,
            ClaimsTags="",
            REFnr="",
            Abstract=abstract,
            LLM_Summary=""  # to be filled later by LLM (~120 words)
        )
    except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError):
        print(f"[WARN] CrossRef lookup failed for {doi}")
        return None


# Helper to create minimal metadata when DOI is missing
def meta_from_pdf(pdf_path: pathlib.Path) -> dict:
    """Create minimal metadata when DOI is missing."""
    name_stub = pdf_path.stem.replace(" ", "_")[:40]
    try:
        reader = PdfReader(str(pdf_path))
        first_pages = (reader.pages[0].extract_text() or "")[:3000]
    except Exception:
        first_pages = ""
    short_name = name_stub[:30]
    abstract = first_pages.replace("\n", " ").strip()
    summary = generate_summary(abstract) if abstract else ""
    return dict(
        ShortName=short_name,
        Year="",
        FirstAuthor="",
        Journal="",
        DOI="N/A",
        ClaimsTags="",
        REFnr="",
        Abstract=abstract,
        LLM_Summary=summary,
    )

def load_index() -> pd.DataFrame:
    cols = ["ShortName","Year","FirstAuthor","Journal",
            "DOI","ClaimsTags","REFnr","Abstract","LLM_Summary"]
    if INDEX_CSV.exists() and INDEX_CSV.stat().st_size > 0:
        try:
            return pd.read_csv(INDEX_CSV, dtype=str).fillna("").reindex(columns=cols)
        except pd.errors.EmptyDataError:
            pass
    return pd.DataFrame(columns=cols)

def main():
    index_df = load_index()
    for pdf in PDF_DIR.glob("*.pdf"):
        doi = extract_doi(pdf)
        if not doi:
            print(f"[INFO] No DOI in {pdf.name} – generating basic entry.")
            meta = meta_from_pdf(pdf)
            index_df = pd.concat([index_df, pd.DataFrame([meta])], ignore_index=True)
            continue
        if (index_df["DOI"] == doi).any():
            continue  # already present
        meta = crossref_meta(doi)
        if not meta:
            continue

        # If Abstract empty, use first 2 pages as surrogate abstract
        if not meta["Abstract"]:
            try:
                reader = PdfReader(str(pdf))
                meta["Abstract"] = (
                    (reader.pages[0].extract_text() or "")[:2000]
                    .replace("\n", " ").strip()
                )
            except Exception:
                pass

        # Always create summary if field is empty
        if not meta["LLM_Summary"]:
            source_text = meta["Abstract"] or " ".join(str(v) for v in meta.values())
            meta["LLM_Summary"] = generate_summary(source_text)

        index_df = pd.concat([index_df, pd.DataFrame([meta])], ignore_index=True)

    index_df.to_csv(INDEX_CSV, index=False)
    print("Index updated. Total rows:", len(index_df))

if __name__ == "__main__":
    main()