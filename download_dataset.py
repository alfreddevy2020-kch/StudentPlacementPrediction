import os
import shutil
from pathlib import Path
import kagglehub

def main():
    # Target directory and file path
    raw_dir = Path("data") / "raw"
    target_path = raw_dir / "student_placement.csv"

    print("Downloading dataset using kagglehub...")
    download_path = kagglehub.dataset_download(
        "suvidyasonawane/student-academic-placement-performance-dataset"
    )
    print(f"Dataset downloaded to cache: {download_path}")

    # Ensure data/raw folder exists
    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded_dir = Path(download_path)
    csv_files = list(downloaded_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in downloaded dataset directory: {download_path}")

    # Prefer student_placement.csv if explicitly named, otherwise use the first CSV found
    src_file = downloaded_dir / "student_placement.csv"
    if not src_file.exists():
        src_file = csv_files[0]

    shutil.copy2(src_file, target_path)
    print(f"[SUCCESS] Saved dataset to {target_path}")

if __name__ == "__main__":
    main()