import csv
from pathlib import Path
from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail


SUPPORTED_EXTS = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")


def process_wsi_folder(input_dir, output_dir, tissue_threshold=0.3):

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    included_dir = output_dir / "included"
    low_dir = output_dir / "low_tissue"

    included_dir.mkdir(parents=True, exist_ok=True)
    low_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "metadata.csv"

    slide_paths = list(input_dir.rglob("*"))
    slide_paths = [p for p in slide_paths if p.suffix.lower() in SUPPORTED_EXTS]

    print("Found {} slides".format(len(slide_paths)))

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "slide_path",
            "thumbnail_path",
            "tissue_fraction",
            "included"
        ])

        for i, slide_path in enumerate(slide_paths):

            print("[{}/{}] {}".format(i + 1, len(slide_paths), slide_path))

            base_name = slide_path.stem + ".jpg"

            # always generate first (no threshold filtering)
            result = create_wsi_thumbnail(
                slide_path,
                output_path=None,  # don't save yet
                tissue_threshold=None,
                save_mask=False,
            )

            if result is None:
                print("Failed\n")

                writer.writerow([
                    str(slide_path),
                    "",
                    "",
                    False
                ])
                continue

            img, tissue_fraction = result

            # decide where to store
            if tissue_fraction < tissue_threshold:
                out_file = low_dir / base_name
                included = False
            else:
                out_file = included_dir / base_name
                included = True

            # save image
            img.save(out_file, quality=90)

            print("Tissue fraction: {:.3f}".format(tissue_fraction))
            print("Saved to: {}\n".format(out_file))

            writer.writerow([
                str(slide_path),
                str(out_file),
                "{:.6f}".format(tissue_fraction),
                included
            ])