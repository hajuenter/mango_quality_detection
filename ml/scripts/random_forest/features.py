import cv2
import numpy as np
from skimage.feature import local_binary_pattern

IMAGE_SIZE = (224, 224)
LBP_P = 8  # jumlah titik tetangga
LBP_R = 1  # radius


def extract_color_features(image_path, size=IMAGE_SIZE):
    """Fitur warna: mean + std RGB + HSV"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")
    img = cv2.resize(img, size)

    # RGB features
    mean_rgb = img.mean(axis=(0, 1))
    std_rgb = img.std(axis=(0, 1))

    # HSV features (lebih robust untuk deteksi warna)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mean_hsv = img_hsv.mean(axis=(0, 1))
    std_hsv = img_hsv.std(axis=(0, 1))

    return np.concatenate([mean_rgb, std_rgb, mean_hsv, std_hsv])


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


def extract_shape_features(image_path, size=IMAGE_SIZE):
    """Fitur bentuk: edge density, contour, dan smoothness"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")
    img = cv2.resize(img, size)

    # Edge detection
    edges = cv2.Canny(img, 50, 150)
    edge_density = edges.sum() / (size[0] * size[1])

    # Contour features
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    num_contours = len(contours)

    # Smoothness (variasi intensitas)
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    smoothness = laplacian.var()

    return np.array([edge_density, num_contours, smoothness])


def extract_statistical_features(image_path, size=IMAGE_SIZE):
    """Fitur statistik tambahan: entropy dan contrast"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")
    img = cv2.resize(img, size)

    # Entropy (ukuran randomness)
    hist, _ = np.histogram(img.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]  # hindari log(0)
    entropy = -np.sum(hist * np.log2(hist))

    # Contrast
    contrast = img.std()

    # Skewness dan Kurtosis
    mean = img.mean()
    std = img.std()
    if std > 0:
        skewness = ((img - mean) ** 3).mean() / (std**3)
        kurtosis = ((img - mean) ** 4).mean() / (std**4)
    else:
        skewness = 0
        kurtosis = 0

    return np.array([entropy, contrast, skewness, kurtosis])


def extract_features(image_path):
    """Gabungkan semua fitur"""
    color = extract_color_features(image_path)  # 12 features (RGB+HSV)
    texture = extract_texture_features(image_path)  # 10 features (LBP)
    shape = extract_shape_features(image_path)  # 3 features
    stats = extract_statistical_features(image_path)  # 4 features

    # Total: 29 features
    return np.concatenate([color, texture, shape, stats])
