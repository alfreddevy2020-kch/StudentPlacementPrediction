import kagglehub

# Download latest version
path = kagglehub.dataset_download(
    "suvidyasonawane/student-academic-placement-performance-dataset"
)

print("Path to dataset files:", path)