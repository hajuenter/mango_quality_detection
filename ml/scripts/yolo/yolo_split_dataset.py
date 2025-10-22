import os
import shutil
import random
from pathlib import Path

source_dir = Path("ml/dataset/mango_dataset_dl")
output_dir = Path("ml/dataset/mango_dataset_dl_split")

classes = ["mango_healthy", "mango_rotten"]

train_ratio = 0.8

for cls in classes:
    imgs = list((source_dir / cls).glob("*.*"))
    random.shuffle(imgs)
    split_point = int(len(imgs) * train_ratio)

    train_imgs = imgs[:split_point]
    val_imgs = imgs[split_point:]

    for mode, files in [("train", train_imgs), ("val", val_imgs)]:
        dest_dir = output_dir / mode / cls
        dest_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            shutil.copy(f, dest_dir / f.name)

print("✅ Dataset successfully split into train/val sets!")
