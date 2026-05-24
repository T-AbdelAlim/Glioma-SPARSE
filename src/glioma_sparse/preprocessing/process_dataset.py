import csv
from pathlib import Path
from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail


SUPPORTED_EXTS = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")


def process_wsi_folder(
    input_dir,
    output_dir,
    tissue_threshold=0.3,
    threshold_metric="tissue"  # "tissue" or "effective"
):

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
            "effective_tissue_fraction",
            "included"
        ])

        for i, slide_path in enumerate(slide_paths):

            print("[{}/{}] {}".format(i + 1, len(slide_paths), slide_path))

            base_name = slide_path.stem + ".jpg"

            result = create_wsi_thumbnail(
                slide_path,
                output_path=None,
                tissue_threshold=None,
                save_mask=False,
            )

            if result is None:
                print("Failed\n")

                writer.writerow([
                    str(slide_path),
                    "",
                    "",
                    "",
                    False
                ])
                continue

            # now returns: img, tissue_fraction, tissue_pixels
            img, tissue_fraction, tissue_pixels = result

            # ----------------------------------------------------
            # Compute effective tissue fraction (correct)
            # ----------------------------------------------------
            padded_area = img.size[0] * img.size[1]
            effective_fraction = tissue_pixels / float(padded_area)

            # ----------------------------------------------------
            # Decide inclusion based on chosen metric
            # ----------------------------------------------------
            if threshold_metric == "effective":
                metric_value = effective_fraction
            else:
                metric_value = tissue_fraction

            if metric_value < tissue_threshold:
                out_file = low_dir / base_name
                included = False
            else:
                out_file = included_dir / base_name
                included = True

            try:
                img.save(out_file, quality=90)

                print("Tissue fraction: {:.2f}".format(tissue_fraction))
                print("Effective fraction: {:.2f}".format(effective_fraction))
                print("Saved to: {}\n".format(out_file))

            except Exception as e:
                print("Save failed: {}\n".format(e))

                writer.writerow([
                    str(slide_path),
                    "",
                    "{:.2f}".format(tissue_fraction),
                    "{:.2f}".format(effective_fraction),
                    False
                ])
                continue

            writer.writerow([
                str(slide_path),
                str(out_file),
                "{:.2f}".format(tissue_fraction),
                "{:.2f}".format(effective_fraction),
                included
            ])