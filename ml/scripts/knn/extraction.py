import os
import pandas as pd
from tqdm import tqdm
from features import extract_features

BASE_SPLIT_DIR = "ml/dataset/mango_dataset_ml_split"

FEATURE_NAMES = [
    "avg_red",
    "avg_green",
    "avg_blue",
    "contrast",
    "homogeneity",
    "correlation",
    "energy",
    "saturation_mean",
    "brightness_mean",
    "entropy",
]


def gather_features(split_dir):
    rows = []
    for split in ["train", "val", "test"]:
        split_path = os.path.join(split_dir, split)
        if not os.path.exists(split_path):
            continue

        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if not os.path.isdir(class_path):
                continue

            for fname in tqdm(os.listdir(class_path), desc=f"{split}/{class_name}"):
                if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                    continue

                img_path = os.path.join(class_path, fname)

                try:
                    feats = extract_features(img_path, split)
                except Exception as e:
                    print("Gagal ekstrak:", img_path, e)
                    continue

                row = {
                    "image_name": fname,
                    "image_path": img_path,
                    "label": class_name,
                    "split": split,
                }

                for feature_name, value in zip(FEATURE_NAMES, feats):
                    row[feature_name] = float(value)

                rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = gather_features(BASE_SPLIT_DIR)

    df["label"] = df["label"].map({"mango_healthy": 0, "mango_rotten": 1})

    os.makedirs("ml/results/knn/features", exist_ok=True)

    df_train_val = df[df["split"].isin(["train", "val"])].reset_index(drop=True)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    with pd.ExcelWriter(
        "ml/results/knn/features/mango_features_all.xlsx", engine="openpyxl"
    ) as writer:
        df_train_val.to_excel(writer, sheet_name="train_val", index=False)
        df_test.to_excel(writer, sheet_name="test", index=False)

    print("Selesai.")
    print("Total Data:", df.shape)
    print("Train+Val:", df_train_val.shape)
    print("Test:", df_test.shape)
