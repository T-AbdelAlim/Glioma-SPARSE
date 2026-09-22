r"""
Same as resolve_radboud_resection_list.py, for biopsy_Radboud.xlsx. Reads
columns by header name since this sheet has an extra "Notitie (onderzoek)"
column that shifts the others.

Usage:
    python -m scripts.analysis.resolve_radboud_biopsy_list
"""

from pathlib import Path
from collections import Counter
import openpyxl

REPO_ROOT = Path(__file__).resolve().parents[2]
RADBOUD_ROOT = Path(r"E:\Thinkpad_Backup\Data\WSI_datasets\Radboud_data_P1000\external_validation_set")
BIOPSY_XLSX = REPO_ROOT / "data" / "Radboud" / "biopsy_Radboud.xlsx"
OUT_PATH = RADBOUD_ROOT / "biopsy_slide_list.txt"


def main():
    wb = openpyxl.load_workbook(BIOPSY_XLSX, data_only=True)
    ws = wb["Sheet1"]
    all_rows = list(ws.iter_rows(values_only=True))
    header, data_rows = all_rows[0], all_rows[1:]
    col = {name: i for i, name in enumerate(header)}
    print(f"columns: {header}")
    print(f"Excel rows (full biopsy list): {len(data_rows)}")

    mrxs_files = list(RADBOUD_ROOT.rglob("*.mrxs"))
    print(f"mrxs files found under {RADBOUD_ROOT}: {len(mrxs_files)}")

    matched, unmatched = [], []
    for row in data_rows:
        rapport, mdn = row[col["RAPPORT:"]], row[col["MDN:"]]
        if not rapport or not mdn:
            unmatched.append(row)
            continue
        hits = [p for p in mrxs_files if mdn in p.stem and rapport in p.stem]
        if hits:
            matched.append((row, hits[0]))
        else:
            unmatched.append(row)

    gunknown = [(r, p) for r, p in matched if "gunknown" in p.parent.name.lower()]
    kept = [(r, p) for r, p in matched if "gunknown" not in p.parent.name.lower()]

    print(f"\nmatched to an actual file: {len(matched)} / {len(data_rows)}")
    print(f"  of which Gunknown (excluded): {len(gunknown)}")
    print(f"  kept (non-Gunknown): {len(kept)}")
    print(f"unmatched (not in this downloaded subset): {len(unmatched)}")

    var_counts = Counter(r[col["Var"]] for r, _ in kept)
    print(f"\nkept, by Excel Var column: {dict(var_counts)}")

    by_folder = Counter(p.parent.name for _, p in kept)
    print("kept, by actual subtype folder:")
    for k, v in sorted(by_folder.items()):
        print(f"  {k}: {v}")

    # cross-check Excel Var vs actual folder, same as the resection resolver
    mismatches = []
    for row, p in kept:
        var = (row[col["Var"]] or "").lower()
        fl = p.parent.name.lower()
        ok = ((var == "gbm" and ("gbm" in fl or "idhwt" in fl)) or
              (var == "oligo" and "oligo" in fl) or
              (var == "astro" and "astro" in fl and "oligo" not in fl))
        if not ok:
            mismatches.append((row, p.name, fl))
    print(f"\nmismatches between Excel Var and actual folder: {len(mismatches)}")
    for m in mismatches[:10]:
        print(" ", m)

    with open(OUT_PATH, "w") as f:
        for _, p in kept:
            f.write(str(p) + "\n")
    print(f"\nWrote {len(kept)} paths to {OUT_PATH}")


if __name__ == "__main__":
    main()
