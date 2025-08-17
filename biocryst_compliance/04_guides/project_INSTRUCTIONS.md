# Project Instructions – HAE Literature Workflow  
*(v 2025‑05‑03 — see changelog at bottom)*

## 📌 Golden Path (30‑second cheat‑sheet)
1. **Default Country = European Union** unless explicitly changed.  
2. Process **1–2 PDFs per batch** → return Veeva table + Vancouver citation.  
3. Cite internal PDFs with the format `citeturnXfileY`.

---

## 1. Veeva Promomats Table – required field order

| Field | Required / Default | Notes |
|-------|--------------------|-------|
| **Name\*** | *auto‑build* → `Surname et al. Short title (Year)` | Keep ≤75 chars; if >3 authors, list only the first surname. |
| Description | *free text* (1–2 sentences; key findings) | |
| Version | **0.1** | Updated by QC team later |
| Type | **Reference** | |
| Subtype | `Clinical Study` \| `Review Article` \| `Guideline` \| … | |
| **Country\*** | **European Union** | Org requirement |
| Product | `Hereditary angioedema (HAE)` | add drug if needed |
| Reference Source | Journal title | |
| Source Date\* | `DD Mon YYYY` (online‑first accepted) | |
| Volume / Issue | `Vol (Issue)` or just `Vol` | |
| Page or Article range | `123‑130` or `Article 100792` | |
| Authors / Data Source\* | `Surname AA; Surname BB` (first two) | |

\* = mandatory in Veeva.

---

## 2. Vancouver‑style reference (+ DOI link)
Surname A, Surname B, Surname C, *et al.* Full article title. Abbrev journal. Year;Volume(Issue):Pages. DOI: xx.xxxx/xxxx – https://doi.org/xx.xxxx/xxxx
*Always include both the DOI and the full `https://doi.org/…` URL.*

---

## 3. Batch policy & timing
| Batch size | Expected turnaround |
|------------|--------------------|
| 1 PDF | 10–15 min |
| 2 PDFs | 15–25 min |
| >2 PDFs | Break into multiple requests |

## 4. Ambiguity rule
If a claim in the draft text is ambiguous or contradictory, request clarification before assigning or creating a source. Do **not** guess or invent citations.

  
## 5. Citation marking inside ChatGPT answers
* Use `citeturnXfileY` immediately after the sentence that relies on the PDF.  
* Multiple PDFs → `citeturn3file0turn4file2`.

---

## 6. Reference validation checklist
* DOI resolves with HTTP 200.  
* Journal name and year match CrossRef metadata.  
* Page range or article number present.  
* If article is ahead‑of‑print, mark as `epub` in Vancouver reference.  
* For SmPCs, verify latest EMA revision date.

---

## 7. Mini‑Changelog
| Date | Version | Change |
|------|---------|--------|
| 2025‑05‑04 | v 2025‑05‑04 | Added name‑rule, ambiguity rule, reference validation checklist; clarified edit approval via PR. |
| 2025‑05‑03 | v 2025‑05‑03 | Initial commit – field order, Vancouver format, batch rules. |