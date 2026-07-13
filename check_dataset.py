from PIL import Image
import os

dataset = "dataset"

bad_files = []

for root, dirs, files in os.walk(dataset):
    for file in files:
        path = os.path.join(root, file)
        try:
            with Image.open(path) as img:
                img.load()
        except Exception as e:
            bad_files.append(path)
            print(f"BAD FILE: {path}")
            print(f"ERROR: {e}")
            print("-" * 50)

if len(bad_files) == 0:
    print("✅ No corrupted images found!")
else:
    print(f"\nTotal bad files: {len(bad_files)}")