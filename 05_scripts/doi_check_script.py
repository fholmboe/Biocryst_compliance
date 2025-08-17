"""
doi_check_script.py – Quick DOI validator

• Läser 00_reference_index.csv.
• Letar kolumnen 'DOI' (skiftlägesoberoende).
• Skickar HEAD‑request mot https://doi.org/<doi>.
• Skriver ut DOI som ger HTTP‑kod ≠ 200 eller fel.
• Kör:  python doi_check_script.py
"""

import csv, pathlib, sys, requests

CSV_PATH = pathlib.Path("00_reference_index.csv")
if not CSV_PATH.exists():
    sys.exit("CSV‑filen saknas")

with CSV_PATH.open(encoding="utf-8") as f:
    reader = csv.DictReader(f)
    doi_key = next(k for k in reader.fieldnames if k.lower() == "doi")
    for row in reader:
        doi = (row.get(doi_key) or "").strip()
        if not doi:
            print(f"[EMPTY DOI] {row}")
            continue
        try:
            r = requests.head(f"https://doi.org/{doi}", allow_redirects=True, timeout=10)
            if r.status_code != 200:
                print(f"{doi}  →  HTTP {r.status_code}")
        except requests.RequestException as err:
            print(f"{doi}  →  ERROR {err.__class__.__name__}")