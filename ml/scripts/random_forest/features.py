import cv2
import numpy as np
from skimage.feature import local_binary_pattern

IMAGE_SIZE = (224, 224)
LBP_P = 8  # jumlah titik tetangga
LBP_R = 1  # radius


def extract_color_features(image_path, size=IMAGE_SIZE):
    """Fitur warna: mean + std RGB"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")
    img = cv2.resize(img, size)
    mean_color = img.mean(axis=(0, 1))
    std_color = img.std(axis=(0, 1))
    return np.concatenate([mean_color, std_color])


def extract_texture_features(image_path, size=IMAGE_SIZE):
    """Fitur tekstur: LBP histogram"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")
    img = cv2.resize(img, size)
    lbp = local_binary_pattern(img, P=LBP_P, R=LBP_R, method="uniform")
    # histogram LBP, normalisasi
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_features(image_path):
    """Gabungkan warna + tekstur"""
    color = extract_color_features(image_path)
    texture = extract_texture_features(image_path)
    return np.concatenate([color, texture])
