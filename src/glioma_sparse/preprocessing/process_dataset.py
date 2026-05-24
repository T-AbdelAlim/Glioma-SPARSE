import csv
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from glioma_sparse.preprocessing.create_wsi_thumbnail import create_wsi_thumbnail


SUPPORTED_EXTS = (".svs", ".ndpi", ".mrxs", ".tif", ".tiff")


# ============================================================
# WORKER
# ============================================================

def _process_single_slide(args):
    slide_path, output_dir, tissue_threshold, threshold_metric = args

    start = time.time()

    included_dir = output_dir / "included"
    low_dir = output_dir / "low_tissue"

    base_name = slide_path.stem + ".jpg"

    try:
        result = create_wsi_thumbnail(
            slide_path,
            output_path=None,
            save_mask=False,
        )

        if result is None:
            raise RuntimeError("Thumbnail creation failed")

        img, tissue_fraction, effective_fraction = result

        # ----------------------------------------------------
        # Decide inclusion
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

        img.save(out_file, quality=90)

        elapsed = time.time() - start

        return {
            "slide_path": str(slide_path),
            "thumbnail_path": str(out_file),
            "tissue_fraction": round(tissue_fraction, 4),
            "effective_tissue_fraction": round(effective_fraction, 4),
            "included": included,
            "processing_time_sec": round(elapsed, 3),
            "success": True,
            "error": ""
        }

    except Exception as e:
        elapsed = time.time() - start

        return {
            "slide_path": str(slide_path),
            "thumbnail_path": "",
            "tissue_fraction": "",
            "effective_tissue_fraction": "",
            "included": False,
            "processing_time_sec": round(elapsed, 3),
            "success": False,
            "error": str(e)
        }


# ============================================================
# MAIN
# ============================================================

def process_wsi_folder(
    input_dir,
    output_dir,
    tissue_threshold=0.3,
    threshold_metric="tissue",
    num_workers=1,   # 👈 backward compatible default
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

    print(f"Found {len(slide_paths)} slides")

    if len(slide_paths) == 0:
        print("No slides found, exiting.")
        return

    t0 = time.time()

    tasks = [
        (p, output_dir, tissue_threshold, threshold_metric)
        for p in slide_paths
    ]

    results = []
    success = 0
    failed = 0

    # ============================================================
    # EXECUTION MODE
    # ============================================================

    if num_workers == 1:
        print("Running sequentially\n")

        for i, task in enumerate(tasks, 1):
            print(f"[{i}/{len(tasks)}] {task[0]}")

            result = _process_single_slide(task)
            results.append(result)

            if result["success"]:
                success += 1
            else:
                failed += 1
                print("FAILED:", result["error"])

    else:
        print(f"Running with {num_workers} workers\n")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_single_slide, t) for t in tasks]

            for i, f in enumerate(as_completed(futures), 1):

                result = f.result()
                results.append(result)

                if result["success"]:
                    success += 1
                else:
                    failed += 1
                    print("FAILED:", result["slide_path"])
                    print("  Error:", result["error"])

                if i % 10 == 0 or i == len(tasks):
                    print(f"[{i}/{len(tasks)}] processed")

    total_time = time.time() - t0

    # ============================================================
    # SAVE CSV
    # ============================================================

    print("\nSaving metadata...")

    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # ============================================================
    # SUMMARY
    # ============================================================

    print("\n=== SUMMARY ===")
    print(f"Total slides: {len(slide_paths)}")
    print(f"Success:      {success}")
    print(f"Failed:       {failed}")
    print(f"Total time:   {total_time:.2f} sec")

    slides_per_sec = success / total_time if total_time > 0 else 0
    slides_per_hour = slides_per_sec * 3600

    print(f"Slides/sec:   {slides_per_sec:.3f}")
    print(f"Slides/hour:  {slides_per_hour:.1f}")

    print(f"\nMetadata saved to: {csv_path}")