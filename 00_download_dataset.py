"""
=============================================================
STEP 0 -- Download Dataset from Kaggle
=============================================================
Dataset : rahuljangir78/recruitment-bias-and-fairness-ai-dataset
Output  : data/recruitment_bias.csv
=============================================================

HOW TO GET YOUR KAGGLE API KEY (one-time setup):
1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section -> click "Create New Token"
3. A file called kaggle.json will download
4. Place it at: C:\Users\<YourName>\.kaggle\kaggle.json
   (Create the .kaggle folder if it doesn't exist)

Then run this script:
    python 00_download_dataset.py
"""

import os
import sys
import shutil

KAGGLE_JSON_PATH = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
OUTPUT_DIR = "data"
DATASET_SLUG = "rahuljangir78/recruitment-bias-and-fairness-ai-dataset"


def check_kaggle_credentials():
    if not os.path.exists(KAGGLE_JSON_PATH):
        print("\n" + "=" * 60)
        print("  KAGGLE API KEY NOT FOUND")
        print("=" * 60)
        print("\nSteps to fix this:")
        print("  1. Go to https://www.kaggle.com/settings")
        print("  2. Scroll to 'API' -> click 'Create New Token'")
        print("  3. Download kaggle.json")
        print(f"  4. Move it to: {KAGGLE_JSON_PATH}")
        print("     (Create the .kaggle folder if needed)")
        print("\nThen re-run: python 00_download_dataset.py")
        print("=" * 60 + "\n")
        sys.exit(1)
    print("[OK] Kaggle credentials found.")


def download_dataset():
    try:
        import kagglehub
        from kagglehub import KaggleDatasetAdapter

        print(f"\n[INFO] Downloading dataset: {DATASET_SLUG}")
        print("[INFO] Please wait...\n")

        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            DATASET_SLUG,
            "",
        )

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(OUTPUT_DIR, "recruitment_bias.csv")
        df.to_csv(out_path, index=False)

        print(f"\n[SUCCESS] Dataset saved to: {out_path}")
        print(f"[INFO] Shape: {df.shape}")
        print(f"\nFirst 5 rows:\n")
        print(df.head().to_string())
        print(f"\nColumn names: {list(df.columns)}")
        return True

    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nTrying alternative download via kaggle CLI...")
        return download_via_cli()


def download_via_cli():
    """Fallback: use kaggle CLI to download the dataset."""
    try:
        import subprocess
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        result = subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", DATASET_SLUG,
                "--unzip",
                "-p", OUTPUT_DIR
            ],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("[SUCCESS] Downloaded via kaggle CLI")
            print(result.stdout)
            # Find the CSV file
            for f in os.listdir(OUTPUT_DIR):
                if f.endswith(".csv"):
                    src = os.path.join(OUTPUT_DIR, f)
                    dst = os.path.join(OUTPUT_DIR, "recruitment_bias.csv")
                    if src != dst:
                        shutil.move(src, dst)
                    print(f"[INFO] CSV saved as: {dst}")
                    return True
        else:
            print(f"[ERROR] CLI download failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"[ERROR] CLI method failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("  RECRUITMENT BIAS DATASET -- DOWNLOADER")
    print("=" * 60)

    check_kaggle_credentials()
    success = download_dataset()

    if success:
        print("\n[NEXT STEP] Run: python 01_eda_bias_check.py")
    else:
        print("\n[MANUAL OPTION] Download the CSV manually from:")
        print("  https://www.kaggle.com/datasets/rahuljangir78/recruitment-bias-and-fairness-ai-dataset")
        print(f"  Place the CSV file at: {os.path.join(OUTPUT_DIR, 'recruitment_bias.csv')}")
