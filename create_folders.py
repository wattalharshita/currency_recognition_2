import os

# List of classes
categories = [
    "100_real", "200_real", "500_real",
    "100_fake", "200_fake", "500_fake"
]

# Base dataset folder
base_dir = "dataset"

# Create folders
for category in categories:
    path = os.path.join(base_dir, category)
    os.makedirs(path, exist_ok=True)
    print(f"Created: {path}")
