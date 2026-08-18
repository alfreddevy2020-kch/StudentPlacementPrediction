import os
import shutil
import kagglehub

# 1. Download latest version of the dataset
print("Downloading dataset from Kaggle via kagglehub...")
downloaded_path = kagglehub.dataset_download(
    "sehaj1104/student-placement-prediction-dataset-2026"
)
print("Path to downloaded files:", downloaded_path)

# 2. Define the target 'data/raw' directory
target_dir = os.path.join("data", "raw")
os.makedirs(target_dir, exist_ok=True)

# 3. Find any CSV file in the downloaded path and copy to data/raw/student_placement.csv
csv_found = False
for filename in os.listdir(downloaded_path):
    if filename.endswith(".csv"):
        src_file = os.path.join(downloaded_path, filename)
        dst_file = os.path.join(target_dir, "student_placement.csv")
        shutil.copy(src_file, dst_file)
        print(f"Copied and placed: {filename} -> {dst_file}")
        csv_found = True
        break

if not csv_found:
    print("ERROR: No CSV dataset file found in the downloaded directory.")
else:
    print("Dataset successfully setup at:", os.path.abspath(dst_file))