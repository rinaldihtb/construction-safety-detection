# 🦺 Construction Safety Detector

<p align="center">
  <a href="https://construction-safety-detector.rinaldihtb.my.id"><img src="https://img.shields.io/badge/Live%20Demo-Open%20App-0ea5e9?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Open live demo"></a>
  <a href="https://github.com/rinaldihtb/construction-safety-detection"><img src="https://img.shields.io/badge/Source%20Code-GitHub-181717?style=for-the-badge&logo=github" alt="GitHub repository"></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/Model-YOLOv12-111827" alt="YOLOv12">
  <img src="https://img.shields.io/badge/UI-Gradio-F97316" alt="Gradio">
  <img src="https://img.shields.io/badge/Task-PPE%20Detection-16A34A" alt="PPE detection">
</p>

> An object-detection web application for identifying construction-site personal protective equipment (PPE) from images. Built with Ultralytics YOLO and Gradio.

## Repository overview

| Field | Details |
| --- | --- |
| **Repository name** | `construction-safety-detection` |
| **Description** | Construction safety and PPE detection powered by YOLO and Gradio. |
| **Live demo** | [construction-safety-detector.rinaldihtb.my.id](https://construction-safety-detector.rinaldihtb.my.id) |
| **Source code** | [github.com/rinaldihtb/construction-safety-detection](https://github.com/rinaldihtb/construction-safety-detection) |
| **Core stack** | Python, Ultralytics YOLO, Gradio, OpenCV, Pillow |
| **Default branch** | `master` |

**Suggested GitHub topics:** `computer-vision` · `construction-safety` · `gradio` · `object-detection` · `personal-protective-equipment` · `ppe-detection` · `python` · `ultralytics` · `yolo` · `yolov12`

## Fitur

- Upload gambar (`jpg`, `jpeg`, `png`, atau `webp`) untuk dianalisis.
- Memakai sample gambar acak apabila opsi sample diaktifkan.
- Menampilkan gambar hasil anotasi serta jumlah objek tiap kelas.
- Live streaming webcam bersifat opsional dan dikendalikan melalui `ENABLE_LIVE_STREAMING`.
- Model lokal: `model/best_construction.pt`.

## Struktur proyek

```text
.
├── app.py                  # Aplikasi Gradio dan inferensi YOLO
├── model/
│   ├── best_construction.pt # Bobot model
│   └── train_model.ipynb    # Notebook pelatihan/eksperimen model
├── model/datasets/          # Dataset YOLO untuk training, validasi, dan pengujian
├── samples/                 # Contoh gambar
├── requirements.txt         # Dependensi Python
├── .env.example             # Template konfigurasi lokal
└── README.md
```

## Menjalankan secara lokal

Kebutuhan minimum: Python 3.11 dan `pip`.

```bash
git clone git@github.com:rinaldihtb/construction-safety-detection.git
cd construction-safety-detection

python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
python app.py
```

Gradio akan mencetak alamat aplikasi lokal di terminal. Hentikan aplikasi dengan `Ctrl+C`.

## Konfigurasi environment

Aplikasi memakai `python-dotenv`, sehingga `app.py` membaca file `.env` di folder proyek.

| Variabel | Nilai yang didukung | Fungsi |
| --- | --- | --- |
| `ENABLE_LIVE_STREAMING` | `true` atau `false` | Menampilkan tab **Live Kamera** hanya saat nilainya `true`. |

Contoh `.env`:

```env
ENABLE_LIVE_STREAMING=false
```

Gunakan `false` pada mesin CPU agar resource tidak terbebani inferensi webcam terus-menerus. Untuk mengaktifkan webcam, ubah menjadi `true` dan restart aplikasi.

## Dataset

Dataset berada di `model/datasets/construction-safety` dalam format YOLO dan berasal dari [Roboflow Universe](https://universe.roboflow.com/personal-project-kej16/construction-safety-gsnvb-jl6el/dataset/1). Lisensinya adalah **CC BY 4.0**.

| Split | Gambar | Label | Kegunaan |
| --- | ---: | ---: | --- |
| `train` | 997 | 997 | Melatih model. |
| `valid` | 119 | 119 | Memantau performa dan early stopping saat training. |
| `test` | 90 | 90 | Evaluasi akhir. |
| **Total** | **1.206** | **1.206** |  |

Konfigurasi [data.yaml](model/datasets/construction-safety/data.yaml) mendefinisikan lima kelas berikut.

| ID | Kelas |
| ---: | --- |
| 0 | `helmet` |
| 1 | `no-helmet` |
| 2 | `no-vest` |
| 3 | `person` |
| 4 | `vest` |

## Tahapan training model

Proses lengkap tersedia pada [model/train_model.ipynb](model/train_model.ipynb) dan dijalankan di Google Colab.

1. Siapkan dataset dan dependensi `ultralytics`.
2. Muat bobot pralatih YOLOv12n, lalu latih model menggunakan konfigurasi pada `data.yaml`.
3. Evaluasi hasil model pada data pengujian dengan metrik mAP.
4. Simpan checkpoint terbaik sebagai [model/best_construction.pt](model/best_construction.pt), lalu uji pada gambar sampel.

## Catatan performa

Inferensi YOLO live streaming membutuhkan CPU/GPU yang cukup. Pada mesin CPU, gunakan `ENABLE_LIVE_STREAMING=false` dan lakukan analisis melalui upload gambar. Pada mesin GPU, aktifkan `true` bila CUDA dan dependensi PyTorch GPU telah dikonfigurasi.
