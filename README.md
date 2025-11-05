# Mango Detection API Specification

API untuk deteksi kondisi mangga (sehat/busuk) menggunakan machine learning dengan Flask backend, Firebase Authentication, dan Firestore database.

---

## Setup Virtual Environment (venv)

Ikuti langkah-langkah berikut untuk menyiapkan environment Python:

### 1. Clone Repository

```bash
git clone https://github.com/hajuenter/mango_quality_detection.git
cd mango_quality_detection
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
```

### 3. Aktivasi Virtual Environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Cara Menjalankan

```bash
cd server/ # Pindah dulu ke direktori server
python main.py # Running
```

---

## Base URL

```

http://127.0.0.1:5000
http://192.168.1.7:5000 # Jika diakses dari HP dalam 1 WiFi atau bisa gunakan IP Adress yang sama

```

## Tech Stack

- **Backend**: Flask
- **Database**: Cloud Firestore
- **Authentication**: Firebase Auth (Email/Password)
- **ML Model**: Image Classification (trained using **Random Forest**)

---

## Authentication

Sebagian besar endpoint dilindungi oleh Firebase Authentication. Gunakan token ID dari Firebase Auth.

### Header Format

```

Authorization: Bearer <firebase_id_token>

```

### Error Response (Unauthorized)

**Status Code**: `401 Unauthorized`

```json
{
  "error": "Authorization header missing"
}
```

---

## Endpoints

### 1. Predict Mango Condition

Deteksi kondisi mangga dari gambar dan simpan hasil ke Firestore.

**Endpoint**: `POST /api/predict`

**Authentication**: Tidak diperlukan

**Content-Type**: `multipart/form-data`

**Request Body**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| file | File | Yes | File gambar (.jpg, .jpeg, .png) |

#### Success Response

**Status Code**: `200 OK`

```json
{
  "success": true,
  "message": "Prediksi berhasil dan disimpan ke Firestore.",
  "result": {
    "label": "mango_healthy",
    "confidence": 0.685882952165624,
    "method": "default_0.5",
    "probabilities": {
      "mango_healthy": 0.685882952165624,
      "mango_rotten": 0.31411704783437583
    },
    "threshold_info": {
      "healthy_implied": 0.5,
      "rotten_threshold": 0.5
    }
  },
  "saved": {
    "label": "mango_healthy",
    "confidence": 0.685882952165624,
    "image_url": "http://127.0.0.1:5000/uploads/He40.jpg",
    "timestamp": "Wed, 05 Nov 2025 14:27:31 GMT",
    "date": "2025-11-05",
    "month": "2025-11",
    "year": "2025",
    "method": "default_0.5",
    "season_name": null,
    "season_status": "none"
  }
}
```

#### Error Response - Missing File

**Status Code**: `400 Bad Request`

```json
{
  "success": false,
  "errors": {
    "file": ["Field may not be null."]
  }
}
```

#### Error Response - Invalid Format

**Status Code**: `400 Bad Request`

```json
{
  "success": false,
  "errors": {
    "file": ["File harus berformat .jpg, .jpeg, atau .png."]
  }
}
```

---

### 2. Get All Detections

Ambil semua data deteksi mangga.

**Endpoint**: `GET /api/detections`

**Authentication**: Required (Firebase Auth)

**Headers**:

```
Authorization: Bearer <firebase_id_token>
```

#### Success Response

**Status Code**: `200 OK`

```json
{
  "success": true,
  "count": 10,
  "detections": [
    {
      "id": "NdQUGiFIyBNw08Sq0crh",
      "label": "mango_healthy",
      "confidence": 0.685882952165624,
      "image_url": "http://127.0.0.1:5000/uploads/He40.jpg",
      "timestamp": "2025-11-05T14:27:31.641887+00:00",
      "date": "2025-11-05",
      "month": "2025-11",
      "year": "2025",
      "method": "default_0.5"
    }
  ]
}
```

---

### 3. Get Healthy Mangoes Only

Ambil data mangga sehat saja.

**Endpoint**: `GET /api/detections/healthy`

**Authentication**: Required (Firebase Auth)

**Headers**:

```
Authorization: Bearer <firebase_id_token>
```

#### Success Response

**Status Code**: `200 OK`

```json
{
  "success": true,
  "count": 5,
  "detections": [
    {
      "id": "NdQUGiFIyBNw08Sq0crh",
      "label": "mango_healthy",
      "confidence": 0.685882952165624,
      "image_url": "http://127.0.0.1:5000/uploads/He40.jpg",
      "timestamp": "2025-11-05T14:27:31.641887+00:00",
      "date": "2025-11-05",
      "month": "2025-11",
      "year": "2025",
      "method": "default_0.5"
    },
    {
      "id": "yS76nmwI3eVNww9mcicm",
      "label": "mango_healthy",
      "confidence": 0.5383522698957919,
      "image_url": "http://127.0.0.1:5000/uploads/He71.jpg",
      "timestamp": "2025-11-05T09:54:21.240608+00:00",
      "date": "2025-11-05",
      "month": "2025-11",
      "year": "2025",
      "method": "default_0.5"
    }
  ]
}
```

---

### 4. Get Rotten Mangoes Only

Ambil data mangga busuk saja.

**Endpoint**: `GET /api/detections/rotten`

**Authentication**: Required (Firebase Auth)

**Headers**:

```
Authorization: Bearer <firebase_id_token>
```

#### Success Response

**Status Code**: `200 OK`

```json
{
  "success": true,
  "count": 5,
  "detections": [
    {
      "id": "iYUVdgwZyr4JaE4GrjjL",
      "label": "mango_rotten",
      "confidence": 0.5398442549019157,
      "image_url": "http://127.0.0.1:5000/uploads/Se7.jpg",
      "timestamp": "2025-11-05T09:53:48.979468+00:00",
      "date": "2025-11-05",
      "month": "2025-11",
      "year": "2025",
      "method": "default_0.5"
    },
    {
      "id": "4a7qu1WY03L84ibT6qAF",
      "label": "mango_rotten",
      "confidence": 0.6781734733982855,
      "image_url": "http://127.0.0.1:5000/uploads/Se69.jpg",
      "timestamp": "2025-11-04T21:44:32.806911+00:00",
      "date": "2025-11-04",
      "month": "2025-11",
      "year": "2025",
      "method": "default_0.5"
    }
  ]
}
```

---

### 5. Get Latest Detections

Ambil 5 data deteksi terbaru (campuran sehat dan busuk).

**Endpoint**: `GET /api/detections/latest`

**Authentication**: Required (Firebase Auth)

**Headers**:

```
Authorization: Bearer <firebase_id_token>
```

#### Success Response

**Status Code**: `200 OK`

```json
{
  "success": true,
  "count": 5,
  "detections": [
    {
      "id": "NdQUGiFIyBNw08Sq0crh",
      "label": "mango_healthy",
      "confidence": 0.685882952165624,
      "image_url": "http://127.0.0.1:5000/uploads/He40.jpg",
      "timestamp": "2025-11-05T14:27:31.641887+00:00",
      "date": "2025-11-05",
      "month": "2025-11",
      "year": "2025",
      "method": "default_0.5"
    },
    {
      "id": "yS76nmwI3eVNww9mcicm",
      "label": "mango_healthy",
      "confidence": 0.5383522698957919,
      "image_url": "http://127.0.0.1:5000/uploads/He71.jpg",
      "timestamp": "2025-11-05T09:54:21.240608+00:00",
      "date": "2025-11-05",
      "month": "2025-11",
      "year": "2025",
      "method": "default_0.5"
    },
    {
      "id": "iYUVdgwZyr4JaE4GrjjL",
      "label": "mango_rotten",
      "confidence": 0.5398442549019157,
      "image_url": "http://127.0.0.1:5000/uploads/Se7.jpg",
      "timestamp": "2025-11-05T09:53:48.979468+00:00",
      "date": "2025-11-05",
      "month": "2025-11",
      "year": "2025",
      "method": "default_0.5"
    }
  ]
}
```

---

### 6. Start Season

Mulai musim panen baru.

**Endpoint**: `POST /api/season/start`

**Authentication**: Required (Firebase Auth)

**Content-Type**: `application/json`

**Headers**:

```
Authorization: Bearer <firebase_id_token>
```

**Request Body**:

```json
{
  "name": "Musim Panen November"
}
```

#### Success Response

**Status Code**: `200 OK`

```json
{
  "success": true,
  "message": "Musim 'Musim Panen Maret' berhasil dimulai.",
  "data": {
    "id": "2025-11-05_14-45-30_Musim_Panen_Maret",
    "name": "Musim Panen Maret",
    "status": "active"
  }
}
```

#### Error Response - Season Already Active

**Status Code**: `400 Bad Request`

```json
{
  "success": false,
  "error": "Musim masih aktif. Hentikan musim yang aktif terlebih dahulu."
}
```

---

### 7. Stop Season

Hentikan musim panen yang aktif.

**Endpoint**: `POST /api/season/stop`

**Authentication**: Required (Firebase Auth)

**Headers**:

```
Authorization: Bearer <firebase_id_token>
```

#### Success Response

**Status Code**: `200 OK`

```json
{
  "success": true,
  "message": "Musim aktif telah dihentikan.",
  "data": {
    "id": null,
    "name": null,
    "status": "none"
  }
}
```

---

### 8. Get Current Season

Cek musim panen yang sedang aktif.

**Endpoint**: `GET /api/season/current`

**Authentication**: Required (Firebase Auth)

**Headers**:

```
Authorization: Bearer <firebase_id_token>
```

#### Success Response (Active Season)

**Status Code**: `200 OK`

```json
{
  "success": true,
  "data": {
    "id": "2025-11-05_14-45-30_Musim_Panen_Maret",
    "name": "Musim Panen Maret",
    "status": "active"
  }
}
```

#### Success Response (No Active Season)

**Status Code**: `200 OK`

```json
{
  "success": true,
  "data": {
    "id": null,
    "name": null,
    "status": "none"
  }
}
```

---

### 9. Get Image

Mengambil gambar hasil deteksi yang sudah diupload.

**Endpoint**: `GET /uploads/{filename}`

**Authentication**: Required (Firebase Auth)

**Headers**:

```
Authorization: Bearer <firebase_id_token>
```

**URL Parameters**:
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| filename | string | Yes | Nama file gambar (contoh: Bc7.jpg) |

#### Success Response

**Status Code**: `200 OK`

**Content-Type**: `image/jpeg` atau `image/png`

Returns: Binary image data

#### Example URL

```
GET http://127.0.0.1:5000/uploads/Bc7.jpg
GET http://127.0.0.1:5000/uploads/He40.jpg
```

#### Error Response - Unauthorized

**Status Code**: `401 Unauthorized`

```json
{
  "error": "Authorization header missing"
}
```

#### Error Response - File Not Found

**Status Code**: `404 Not Found`

```json
{
  "error": "File not found"
}
```

---

## Notes

1. **Image Upload**: Endpoint `/api/predict` hanya menerima file gambar dengan format `.jpg`, `.jpeg`, atau `.png`.
2. **Firebase Auth**: Token ID Firebase harus valid dan belum kadaluarsa untuk mengakses endpoint yang dilindungi.
3. **Image Access**:
   - Semua gambar di folder `/uploads/` dilindungi oleh Firebase Auth.
   - Untuk mengakses gambar, gunakan endpoint `GET /uploads/{filename}` dengan header Authorization.
   - Image URLs yang dikembalikan oleh API dapat digunakan langsung dengan menambahkan header Authorization.
4. **Season Management**:
   - Hanya satu musim yang bisa aktif pada satu waktu.
   - Tidak bisa memulai musim baru jika masih ada musim yang aktif.
   - Harus menghentikan musim aktif terlebih dahulu sebelum memulai musim baru.
5. **Timestamps**: Semua timestamp menggunakan format ISO8601 dengan timezone UTC.

## Image Preprocessing & Feature Extraction

Model melakukan ekstraksi fitur dari gambar mangga sebelum proses klasifikasi menggunakan **Random Forest**.  
Tahapan preprocessing dan jenis fitur yang digunakan adalah sebagai berikut:

### **1. Resize Gambar**

Semua gambar diubah ukurannya menjadi **224x224 piksel** agar konsisten dalam proses ekstraksi.

### **2. Color Features (RGB + HSV)**

- Mengambil **mean** dan **standard deviation** dari setiap channel RGB.
- Mengonversi gambar ke **HSV** dan melakukan hal yang sama.
- Total: **12 fitur warna**.

### **3. Texture Features (LBP - Local Binary Pattern)**

- Menggunakan `local_binary_pattern` dari skimage.
- Parameter:
  - `LBP_P = 8` (jumlah titik tetangga)
  - `LBP_R = 1` (radius)
- Histogram LBP digunakan sebagai fitur tekstur.
- Total: **~10 fitur tekstur** (disesuaikan histogram).

### **4. Shape Features**

- **Edge Density** → Menggunakan Canny edge detector.
- **Contour Count** → Jumlah kontur utama.
- **Smoothness** → Variansi Laplacian.
- Total: **3 fitur bentuk**.

### **5. Statistical Features**

- **Entropy** → Tingkat randomness citra.
- **Contrast** → Standard deviation intensitas.
- **Skewness & Kurtosis** → Distribusi tekstur permukaan.
- Total: **4 fitur statistik**.

### **Total Fitur per Gambar**

**29 fitur** yang dihasilkan dari kombinasi:

- 12 fitur warna (RGB + HSV)
- 10 fitur tekstur (LBP histogram)
- 3 fitur bentuk (edge density, contour count, smoothness)
- 4 fitur statistik (entropy, contrast, skewness, kurtosis)

---

## Model Training

Setelah fitur diekstraksi, data dilatih menggunakan algoritma **Random Forest Classifier**.

### **Hyperparameters**

- `n_estimators`: Jumlah decision trees dalam forest
- `max_depth`: Kedalaman maksimal setiap tree
- `min_samples_split`: Jumlah minimum sampel untuk split node
- `random_state`: Seed untuk reproduksibilitas

### **Training Pipeline**

1. Load dataset gambar dari folder terorganisir per kelas
2. Ekstraksi fitur untuk setiap gambar
3. Split data menjadi train, val, dan test
4. Train Random Forest model
5. Evaluasi performa model

---

## Model Evaluation

Evaluasi dilakukan menggunakan metrik berikut:

- **Accuracy**: Persentase prediksi benar
- **Precision**: Ketepatan prediksi positif
- **Recall**: Kemampuan mendeteksi kelas positif
- **F1-Score**: Harmonic mean dari precision dan recall
- **Confusion Matrix**: Visualisasi performa per kelas
