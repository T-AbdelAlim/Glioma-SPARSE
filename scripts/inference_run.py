from pathlib import Path

from scripts.inference_stageA import run_inference


def main():

    # --------------------------------------------------
    # CONFIG (edit freely)
    # --------------------------------------------------
    input_path = Path(
        r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\data\ebrains_thumbnails\control\included"
    )

    checkpoint_path = Path(
        r"C:\Users\Tareq\pythonProject\Glioma-SPARSE\training_output\20260526_0947_resnet34_cw\best_auc.pth"
    )

    model_name = "resnet34"

    # --------------------------------------------------
    # STAGE A
    # --------------------------------------------------
    print("\n=== STAGE A ===\n")

    stageA_df = run_inference(
        input_path=input_path,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
    )

    # --------------------------------------------------
    # FUTURE: STAGE B
    # --------------------------------------------------
    # stageB_df = run_stageB(stageA_df)
    # final_df = combine(stageA_df, stageB_df)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()