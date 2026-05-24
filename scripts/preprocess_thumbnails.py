from pathlib import Path
import argparse

from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail


SUPPORTED_EXTS = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs")


def find_wsi_files(root: Path):
    return [p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS]


def process_folder(input_dir: Path, output_dir: Path):
    files = find_wsi_files(input_dir)

    print(f"\n🔍 Found {len(files)} WSI files\n")

    if not files:
        print("❌ No WSI files found.")
        return

    for i, slide_path in enumerate(files):
        print(f"[{i+1}/{len(files)}] {slide_path}")

        name = slide_path.stem
        out_path = output_dir / f"{name}.jpg"

        try:
            create_wsi_thumbnail(slide_path, out_path)
            print(f"✅ Saved: {out_path}")
        except Exception as e:
            print(f"❌ Failed: {slide_path} ({e})")


def main():
    parser = argparse.ArgumentParser(description="Generate WSI thumbnails")
    parser.add_argument("input_dir", type=Path, help="Input folder with WSIs")
    parser.add_argument("output_dir", type=Path, help="Output folder")

    args = parser.parse_args()

    process_folder(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()