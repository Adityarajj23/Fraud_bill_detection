# split_val.py
import os, random, shutil

SEED = 42
VAL_RATIO = 0.20

BASE = "dataset"
TRAIN_REAL = os.path.join(BASE, "train", "real")
TRAIN_FAKE = os.path.join(BASE, "train", "fake")
VAL_REAL = os.path.join(BASE, "val", "real")
VAL_FAKE = os.path.join(BASE, "val", "fake")

os.makedirs(VAL_REAL, exist_ok=True)
os.makedirs(VAL_FAKE, exist_ok=True)

def list_images(folder):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [f for f in os.listdir(folder) if f.lower().endswith(exts)]

def move_sample(src_dir, dst_dir, ratio):
    files = list_images(src_dir)
    random.Random(SEED).shuffle(files)
    k = max(1, int(len(files) * ratio))
    val_files = files[:k]
    for f in val_files:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        shutil.move(src, dst)
    return len(val_files), len(files) - k

moved_real, remain_real = move_sample(TRAIN_REAL, VAL_REAL, VAL_RATIO)
moved_fake, remain_fake = move_sample(TRAIN_FAKE, VAL_FAKE, VAL_RATIO)

print(f"Validation moved: real={moved_real}, fake={moved_fake}")
print(f"Train remaining: real={remain_real}, fake={remain_fake}")
print("Done. Train/Val split created without touching test_images/.")
