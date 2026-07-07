from pathlib import Path

from scripts.inference_stageA import run_inference


def main():

    # --------------------------------------------------
    # CONFIG (edit freely)
    # --------------------------------------------------
    input_path = Path(
        r"..\data\ebrains_thumbnails\control\included"
    )

    checkpoint_path = Path(
        r"..\training_output\20260703_0928_resnet18_cw_split_03\best_f1.pth"
    )

    model_name = "resnet18"

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