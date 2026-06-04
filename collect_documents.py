"""
Collects 35 legal documents from public HuggingFace datasets:
  - 20 commercial contracts  : dvgodoy/CUAD_v1_Contract_Understanding_PDF
  - 15 Indian SC judgments   : rishiai/indian-court-judgements-and-its-summaries
                               (falls back to Exploration-Lab/IL-TUR if needed)

Saves all docs as .txt files in documents/ and writes documents/metadata.json.
"""

import json
import re
from pathlib import Path
from datasets import load_dataset

DOCS_DIR = Path("documents")
DOCS_DIR.mkdir(exist_ok=True)

metadata = []


def clean_text(text):
    return re.sub(r'\s+', ' ', str(text)).strip()


def sanitize_filename(name):
    name = re.sub(r'[^\w\s-]', '', str(name))
    return re.sub(r'\s+', '_', name.strip())[:75]


# ─── Part 1: CUAD Commercial Contracts ───────────────────────────────────────
print("=" * 60)
print("Part 1: Downloading CUAD contracts from HuggingFace...")
print("=" * 60)

cuad = load_dataset("dvgodoy/CUAD_v1_Contract_Understanding_PDF", split="train")
print(f"  Columns  : {cuad.column_names}")
print(f"  Total    : {len(cuad)} contracts")

text_col = next((c for c in ["text", "content", "contract_text"] if c in cuad.column_names), cuad.column_names[0])
name_col = next((c for c in ["file_name", "title", "name", "id"] if c in cuad.column_names), None)
print(f"  text_col='{text_col}', name_col='{name_col}'\n")

step = max(1, len(cuad) // 20)
selected_indices = list(range(0, min(len(cuad), step * 20), step))[:20]

for i, idx in enumerate(selected_indices):
    row = cuad[idx]
    text = clean_text(row[text_col])
    if len(text) < 500:
        continue

    title = str(row[name_col]).replace(".pdf", "").replace(".txt", "") if name_col else f"Contract_{idx}"
    fname = f"contract_{i+1:02d}_{sanitize_filename(title)}.txt"
    (DOCS_DIR / fname).write_text(text, encoding="utf-8")
    metadata.append({"filename": fname, "title": title, "type": "contract",
                      "source": "CUAD / TheAtticusProject", "char_count": len(text)})
    print(f"  [{i+1:02d}/20] {fname[:70]} ({len(text):,} chars)")

contracts_saved = len([m for m in metadata if m["type"] == "contract"])
print(f"\nContracts done. {contracts_saved}/20 saved.\n")


# ─── Part 2: Indian Supreme Court Judgments ──────────────────────────────────
print("=" * 60)
print("Part 2: Downloading Indian SC judgments from HuggingFace...")
print("=" * 60)

judgments_saved = []

# Primary source
try:
    ds = load_dataset("rishiai/indian-court-judgements-and-its-summaries", split="train")
    print(f"  Columns : {ds.column_names}")
    print(f"  Total   : {len(ds)} judgments")

    col_lower = {c.lower(): c for c in ds.column_names}
    text_col = next(
        (col_lower[k] for k in ["judgment", "text", "content", "judgment_text", "full_text", "body"] if k in col_lower),
        ds.column_names[0]
    )
    title_col = next(
        (col_lower[k] for k in ["title", "case_name", "name", "case_title", "id", "case_no"] if k in col_lower),
        None
    )
    print(f"  text_col='{text_col}', title_col='{title_col}'\n")

    count = 0
    for i, row in enumerate(ds):
        if count >= 15:
            break
        text = clean_text(row[text_col])
        if len(text) < 1000:
            continue
        title = clean_text(row[title_col]) if title_col else f"SC_Judgment_{i+1}"
        fname = f"judgment_{count+1:02d}_{sanitize_filename(title)}.txt"
        (DOCS_DIR / fname).write_text(text, encoding="utf-8")
        judgments_saved.append({"filename": fname, "title": title, "type": "judgment",
                                 "source": "Indian SC Judgments / rishiai HuggingFace", "char_count": len(text)})
        print(f"  [{count+1:02d}/15] {fname[:70]} ({len(text):,} chars)")
        count += 1

except Exception as e:
    print(f"  Primary source failed: {e}")

# Fallback source
if len(judgments_saved) < 15:
    needed = 15 - len(judgments_saved)
    print(f"\n  Fallback: Exploration-Lab/IL-TUR (need {needed} more)...")
    try:
        ds2 = load_dataset("Exploration-Lab/IL-TUR", split="train")
        col_lower2 = {c.lower(): c for c in ds2.column_names}
        text_col2 = next(
            (col_lower2[k] for k in ["text", "judgment", "content", "document"] if k in col_lower2),
            ds2.column_names[0]
        )
        title_col2 = next(
            (col_lower2[k] for k in ["id", "case_id", "title", "name"] if k in col_lower2),
            None
        )
        count = len(judgments_saved)
        for i, row in enumerate(ds2):
            if count >= 15:
                break
            text = clean_text(row[text_col2])
            if len(text) < 1000:
                continue
            title = clean_text(row[title_col2]) if title_col2 else f"SC_Case_{i+1}"
            fname = f"judgment_{count+1:02d}_{sanitize_filename(title)}.txt"
            (DOCS_DIR / fname).write_text(text, encoding="utf-8")
            judgments_saved.append({"filename": fname, "title": title, "type": "judgment",
                                     "source": "Indian SC Cases / IL-TUR HuggingFace", "char_count": len(text)})
            print(f"  [{count+1:02d}/15] {fname[:70]} ({len(text):,} chars)")
            count += 1
    except Exception as e:
        print(f"  Fallback also failed: {e}")

metadata.extend(judgments_saved)
print(f"\nJudgments done. {len(judgments_saved)}/15 saved.\n")


# ─── Save Metadata ────────────────────────────────────────────────────────────
meta_path = DOCS_DIR / "metadata.json"
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("=" * 60)
print(f"COLLECTION COMPLETE")
print(f"  Contracts : {len([m for m in metadata if m['type'] == 'contract'])}")
print(f"  Judgments : {len([m for m in metadata if m['type'] == 'judgment'])}")
print(f"  Total     : {len(metadata)} documents")
print(f"  Metadata  : {meta_path}")
print("=" * 60)
