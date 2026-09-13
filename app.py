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

PPE_LABELS = {"helmet", "no-helmet", "vest", "no-vest"}


def get_box_center(box):
    """Mengambil titik tengah dari sebuah kotak deteksi."""
    left, top, right, bottom = box
    return (left + right) / 2, (top + bottom) / 2


def find_person_for_ppe(ppe_box, people):
    """Mencari person yang paling dekat dan memuat posisi APD."""
    ppe_center_x, ppe_center_y = get_box_center(ppe_box)
    matching_people = []

    for person in people:
        left, top, right, bottom = person["box"]
        is_inside_person = (
            left <= ppe_center_x <= right and top <= ppe_center_y <= bottom
        )
        if is_inside_person:
            matching_people.append(person)

    if not matching_people:
        return None

    # Jika ada kotak person yang tumpang tindih, pilih yang titik tengahnya paling dekat.
    return min(
        matching_people,
        key=lambda person: sum(
            (person_center - ppe_center) ** 2
            for person_center, ppe_center in zip(
                get_box_center(person["box"]),
                (ppe_center_x, ppe_center_y),
            )
        ),
    )


def get_person_safety_status(person):
    """Membuat status kelengkapan APD untuk satu person."""
    missing_items = []
    review_items = []

    if person["no-helmet"]:
        missing_items.append("helmet")
    elif not person["helmet"]:
        review_items.append("helmet tidak terdeteksi")

    if person["no-vest"]:
        missing_items.append("vest")
    elif not person["vest"]:
        review_items.append("vest tidak terdeteksi")

    if missing_items:
        return "Belum lengkap", f"Tidak memakai {', '.join(missing_items)}."
    if review_items:
        return "Perlu diperiksa", f"{'; '.join(review_items).capitalize()}."
    return "Lengkap", "Helmet dan vest terdeteksi."


def build_safety_analysis(detection_result):
    """Menghitung objek dan membuat analisis kelengkapan APD per person."""
    if detection_result.boxes is None:
        return "### Analisis Objek\n\n- Tidak ada objek terdeteksi."

    class_names = detection_result.names
    boxes = detection_result.boxes.xyxy.cpu().numpy()
    class_ids = detection_result.boxes.cls.cpu().numpy().astype(int)
    counts = {}
    people = []
    ppe_items = []

    # Pisahkan hasil deteksi menjadi person dan APD agar dapat dipasangkan.
    for box, class_id in zip(boxes, class_ids):
        class_name = class_names[class_id]
        counts[class_name] = counts.get(class_name, 0) + 1
        item = {"box": box, "label": class_name}

        if class_name == "person":
            people.append(
                {
                    "box": box,
                    "helmet": [],
                    "no-helmet": [],
                    "vest": [],
                    "no-vest": [],
                }
            )
        elif class_name in PPE_LABELS:
            ppe_items.append(item)

    # Pasangkan APD ke person berdasarkan posisi tengah kotak APD.
    for item in ppe_items:
        matched_person = find_person_for_ppe(item["box"], people)
        if matched_person is not None:
            matched_person[item["label"]].append(item)

    lines = ["### Analisis Objek", ""]
    for class_name, count in sorted(counts.items()):
        lines.append(f"- {class_name}: {count}")

    if people:
        lines.extend(["", "### Kelengkapan APD per Pekerja", ""])
        complete_count = 0
        incomplete_count = 0
        review_count = 0

        for index, person in enumerate(people, start=1):
            status, detail = get_person_safety_status(person)
            lines.append(f"- Pekerja {index}: **{status}** — {detail}")

            if status == "Lengkap":
                complete_count += 1
            elif status == "Belum lengkap":
                incomplete_count += 1
            else:
                review_count += 1

        lines.extend(
            [
                "",
                "### Simpulan",
                "",
                f"Terdeteksi **{len(people)} pekerja** pada gambar.",
            ]
        )

        if complete_count:
            lines.append(
                f"- **{complete_count} pekerja** terlihat memakai helmet dan vest."
            )
        if incomplete_count:
            lines.append(
                f"- **{incomplete_count} pekerja** terlihat belum memakai APD dengan lengkap."
            )
        if review_count:
            lines.append(
                f"- **{review_count} pekerja** perlu diperiksa lagi karena APD-nya belum terdeteksi dengan jelas."
            )

        lines.extend(
            [
                "",
                "Catatan: sistem memasangkan APD ke pekerja berdasarkan posisi kotak deteksi pada gambar.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "### Simpulan",
                "",
                "Tidak ada pekerja yang terdeteksi pada gambar.",
            ]
        )

    return "\n".join(lines)


def darken_outside_boxes(original_image, detection_result):
    """Membuat area di luar kotak deteksi terlihat lebih gelap."""
    # Buat mask hitam seukuran gambar.
    detected_object_area = np.zeros(original_image.shape[:2], dtype=np.uint8)

    # Tandai isi setiap kotak deteksi dengan warna putih.
    if detection_result.boxes is not None:
        for left, top, right, bottom in (
            detection_result.boxes.xyxy.cpu().numpy().astype(int)
        ):
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


def analyze_image(image, use_random_sample):
    """Menganalisis gambar upload atau contoh acak, lalu menampilkan ringkasannya."""
    # Jika belum ada gambar upload, pakai contoh acak bila opsinya dicentang.
    if image is None:
        if not use_random_sample:
            return None, "Silakan upload gambar atau centang opsi sample acak."
        if not sample_images:
            return None, "Silakan upload gambar terlebih dahulu."
        image = Image.open(random.choice(sample_images)).convert("RGB")

    # Deteksi objek pada gambar, lalu buat analisis hasilnya.
    result_image, detection_result = detect_image(image)
    analysis = build_safety_analysis(detection_result)

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
