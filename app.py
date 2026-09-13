# Baca pengaturan aplikasi dan siapkan gambar contoh.
import os
import random
from pathlib import Path
from typing import cast

# Library untuk deteksi gambar dan membuat tampilan web.
import cv2
import gradio as gr
import numpy as np
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results

# Ambil pengaturan dari .env, misalnya untuk menyalakan live camera.
load_dotenv()
ENABLE_LIVE_STREAMING = os.getenv("ENABLE_LIVE_STREAMING", "false").lower() == "true"

# Model hasil training ini dipakai untuk mendeteksi objek pada gambar.
model = YOLO("./model/best_construction.pt")

# Siapkan daftar gambar yang bisa dipakai sebagai contoh acak.
SAMPLE_DIR = Path("samples")
sample_images = sorted(
    [
        path
        for path in SAMPLE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
)


def darken_outside_boxes(original_image, detection_result):
    """Membuat area di luar kotak deteksi terlihat lebih gelap."""
    # Buat mask hitam seukuran gambar.
    detected_object_area = np.zeros(original_image.shape[:2], dtype=np.uint8)

    # Tandai isi setiap kotak deteksi dengan warna putih.
    if detection_result.boxes is not None:
        for left, top, right, bottom in detection_result.boxes.xyxy.cpu().numpy().astype(int):
            cv2.rectangle(
                detected_object_area,
                (left, top),
                (right, bottom),
                255,
                thickness=-1,
            )

    # Gelapkan gambar, lalu kembalikan bagian dalam kotak seperti semula.
    dimmed_image = cv2.convertScaleAbs(original_image, alpha=0.45, beta=0)
    # dimmed_image[detected_object_area == 255] = original_image[
    #     detected_object_area == 255
    # ]
    return dimmed_image


def detect_image(image, image_size=640):
    """Mendeteksi objek dan membuat gambar dengan kotak serta label."""
    prediction_results = model.predict(image, imgsz=image_size, verbose=False)
    detection_result = cast(Results, next(iter(prediction_results)))

    # Gelapkan gambar asli dulu agar kotak dan label tetap terlihat jelas.
    dimmed_image = darken_outside_boxes(
        detection_result.orig_img.copy(), detection_result
    )

    annotated_image = detection_result.plot(img=dimmed_image)
    display_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(display_image), detection_result

    # return Image.fromarray(result_image), result


def analyze_image(image, use_random_sample):
    """Menganalisis gambar upload atau contoh acak, lalu menampilkan ringkasannya."""
    # Jika belum ada gambar upload, pakai contoh acak bila opsinya dicentang.
    if image is None:
        if not use_random_sample:
            return None, "Silakan upload gambar atau centang opsi sample acak."
        if not sample_images:
            return None, "Silakan upload gambar terlebih dahulu."
        image = Image.open(random.choice(sample_images)).convert("RGB")

    # Deteksi objek pada gambar, lalu hitung jumlah setiap kelas.
    result_image, detection_result = detect_image(image)
    class_names = detection_result.names # {0: 'helmet', 1: 'no-helmet', 2: 'no-vest', 3: 'person', 4: 'vest'}
    counts = {}

    # Jika tidak ada objek, bagian ini akan dilewati.
    if detection_result.boxes is not None:
        for box in detection_result.boxes:
            class_id = int(box.cls.item())
            class_name = class_names[class_id]
            counts[class_name] = counts.get(class_name, 0) + 1

    # Buat ringkasan singkat dari objek yang ditemukan.
    if counts:
        lines = ["### Analisis Objek", ""]
        total_objects = sum(counts.values())
        for class_name, count in sorted(counts.items()):
            lines.append(f"- {class_name}: {count}")
        lines.extend(
            [
                "",
                "### Simpulan",
                "",
                f"Total objek terdeteksi: **{total_objects}**.",
                "Hasil ini menunjukkan objek yang ditemukan pada gambar upload.",
            ]
        )
        analysis = "\n".join(lines)
    else:
        analysis = (
            "### Analisis Objek\n\n"
            "- Tidak ada objek terdeteksi.\n\n"
            "### Simpulan\n\n"
            "Gambar tidak menunjukkan objek yang dikenali model."
        )

    return result_image, analysis


def detect_camera_frame(frame):
    """Mendeteksi objek dari satu frame kamera dengan ukuran lebih kecil."""
    result_image, _ = detect_image(frame, image_size=416)
    return result_image


def start_app():
    """Membuat lalu menjalankan tampilan aplikasi."""
    with gr.Blocks() as demo:
        # Tampilkan judul dan penjelasan singkat aplikasi.
        gr.Markdown("## Construction Safety Detector")
        gr.Markdown("Upload gambar untuk analisis lengkap.")

        # Tab untuk upload gambar dan melihat hasil deteksinya.
        with gr.Tab("Upload File"):
            with gr.Row():
                # Bagian kiri untuk memilih gambar dan memulai analisis.
                with gr.Column():
                    upload_input = gr.Image(
                        type="pil", sources=["upload"], label="Upload Image"
                    )
                    use_random_sample = gr.Checkbox(
                        label="Gunakan sample acak jika tidak ada gambar",
                        value=False,
                    )
                    upload_btn = gr.Button("Analisis Gambar")

                # Bagian kanan untuk gambar hasil dan ringkasan objek.
                with gr.Column():
                    upload_output = gr.Image(type="pil", label="Detection Result")
                    upload_analysis = gr.Markdown()

            # Jalankan analisis saat tombol ditekan.
            upload_btn.click(
                fn=analyze_image,
                inputs=[upload_input, use_random_sample],
                outputs=[upload_output, upload_analysis],
            )

        # Tab kamera hanya muncul jika diaktifkan dari file .env.
        if ENABLE_LIVE_STREAMING:
            with gr.Tab("Live Kamera"):
                with gr.Row():
                    # Bagian kiri menerima gambar langsung dari kamera.
                    with gr.Column():
                        live_input = gr.Image(
                            type="pil",
                            sources="webcam",
                            streaming=True,
                            label="Live Camera",
                        )

                    # Bagian kanan menampilkan gambar hasil deteksi.
                    with gr.Column():
                        live_output = gr.Image(type="pil", label="Detection Result")

                # Jalankan deteksi selama kamera mengirim gambar.
                live_input.stream(
                    fn=detect_camera_frame,
                    inputs=live_input,
                    outputs=live_output,
                    time_limit=30,
                    stream_every=0.5,
                )

    demo.launch(server_name="0.0.0.0")


# Jalankan aplikasi jika file ini dipanggil secara langsung.
if __name__ == "__main__":
    start_app()
