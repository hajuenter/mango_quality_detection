import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy

IMAGE_SIZE = (224, 224)


def get_avg_rgb(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    avg_rgb = np.mean(rgb, axis=(0, 1))
    return avg_rgb


def get_glcm_features(image):
    """Hitung fitur tekstur dari GLCM (Gray-Level Co-occurrence Matrix)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE)

    glcm = graycomatrix(gray, distances=[1], angles=[0], symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]

    return np.array([contrast, homogeneity, correlation, energy])


def get_hsv_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return np.mean(s), np.mean(v)


def get_entropy_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMAGE_SIZE)
    return shannon_entropy(gray)


def extract_features(image_path, split=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Gagal membaca gambar: {image_path}")

    avg_r, avg_g, avg_b = get_avg_rgb(img)
    glcm_features = get_glcm_features(img)
    sat_mean, val_mean = get_hsv_features(img)
    entropy = get_entropy_feature(img)

    return np.concatenate(
        [[avg_r, avg_g, avg_b], glcm_features, [sat_mean, val_mean, entropy]]
    )
