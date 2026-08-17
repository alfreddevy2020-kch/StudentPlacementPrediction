import os
import shutil
import kagglehub

# 1. Download the latest version of the dataset
downloaded_path = kagglehub.dataset_download(
    "suvidyasonawane/student-academic-placement-performance-dataset"
)

# 2. Define the target 'data/raw' directory
target_dir = os.path.join("data", "raw")
os.makedirs(target_dir, exist_ok=True)

# 3. Find any CSV file and copy/rename it to student_placement.csv in data/raw
csv_found = False
for filename in os.listdir(downloaded_path):
    if filename.endswith(".csv"):
        src_file = os.path.join(downloaded_path, filename)
        dst_file = os.path.join(target_dir, "student_placement.csv")
        shutil.copy(src_file, dst_file)
        print(f"Copied and renamed {filename} -> {dst_file}")
        csv_found = True
        break

if not csv_found:
    print("Warning: No CSV dataset file found in the downloaded directory.")
else:
    print("Dataset files are now stored in:", os.path.abspath(target_dir))
