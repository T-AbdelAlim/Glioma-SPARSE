r"""
Match resection_Radboud.xlsx against the downloaded .mrxs files (patient id
+ report id both in the filename) and write the matched, non-Gunknown paths
for --slide-list filtering.

Usage:
    python -m scripts.analysis.resolve_radboud_resection_list
"""

from pathlib import Path
from collections import Counter
import openpyxl

RADBOUD_ROOT = Path(r"E:\Thinkpad_Backup\Data\WSI_datasets\Radboud_data_P1000\external_validation_set")
RESECTION_XLSX = RADBOUD_ROOT / "resection_Radboud.xlsx"
OUT_PATH = RADBOUD_ROOT / "resection_slide_list.txt"


def main():
    wb = openpyxl.load_workbook(RESECTION_XLSX, data_only=True)
    rows = list(wb["Sheet1"].iter_rows(values_only=True))[1:]  # skip header
    print(f"Excel rows (full resection list): {len(rows)}")

    mrxs_files = list(RADBOUD_ROOT.rglob("*.mrxs"))
    print(f"mrxs files found under {RADBOUD_ROOT}: {len(mrxs_files)}")

    matched, unmatched = [], []
    for row in rows:
        rapport, mdn = row[0], row[1]
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

    print(f"\nmatched to an actual file: {len(matched)} / {len(rows)}")
    print(f"  of which Gunknown (excluded): {len(gunknown)}")
    print(f"  kept (non-Gunknown): {len(kept)}")
    print(f"unmatched (not in this downloaded subset): {len(unmatched)}")

    by_folder = Counter(p.parent.name for _, p in kept)
    print("\nkept, by subtype folder:")
    for k, v in sorted(by_folder.items()):
        print(f"  {k}: {v}")

    with open(OUT_PATH, "w") as f:
        for _, p in kept:
            f.write(str(p) + "\n")
    print(f"\nWrote {len(kept)} paths to {OUT_PATH}")


if __name__ == "__main__":
    main()
