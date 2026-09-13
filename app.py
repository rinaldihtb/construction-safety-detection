import os
import random
from pathlib import Path

import cv2
import gradio as gr
from dotenv import load_dotenv
from PIL import Image
from ultralytics import YOLO

load_dotenv()
ENABLE_LIVE_STREAMING = os.getenv("ENABLE_LIVE_STREAMING", "false").lower() == "true"

model = YOLO("./model/best_construction.pt")
SAMPLE_DIR = Path("samples")
SAMPLE_FILES = sorted(
    [
        path
        for path in SAMPLE_DIR.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]
)


def _annotate(image, imgsz=640):
    result = model.predict(image, imgsz=imgsz, verbose=False)[0]
    annotated = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated), result


def analyze_upload(image, use_random_sample):
    if image is None:
        if not use_random_sample:
            return None, "Silakan upload gambar atau centang opsi sample acak."
        if not SAMPLE_FILES:
            return None, "Silakan upload gambar terlebih dahulu."
        image = Image.open(random.choice(SAMPLE_FILES)).convert("RGB")

    annotated, result = _annotate(image)
    names = result.names
    counts = {}

    for box in result.boxes:
        class_id = int(box.cls.item())
        label = names[class_id]
        counts[label] = counts.get(label, 0) + 1

    if counts:
        lines = ["### Analisis Objek", ""]
        total = sum(counts.values())
        for label, count in sorted(counts.items()):
            lines.append(f"- {label}: {count}")
        lines.extend(
            [
                "",
                "### Simpulan",
                "",
                f"Total objek terdeteksi: **{total}**.",
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

    return annotated, analysis


def live_stream(frame):
    annotated, _ = _annotate(frame, imgsz=416)
    return annotated


def run():
    with gr.Blocks() as demo:
        gr.Markdown("## Construction Safety Detector")
        gr.Markdown("Upload gambar untuk analisis lengkap.")

        with gr.Tab("Upload File"):
            with gr.Row():
                with gr.Column():
                    upload_input = gr.Image(
                        type="pil", sources=["upload"], label="Upload Image"
                    )
                    use_random_sample = gr.Checkbox(
                        label="Gunakan sample acak jika tidak ada gambar",
                        value=False,
                    )
                    upload_btn = gr.Button("Analisis Gambar")
                with gr.Column():
                    upload_output = gr.Image(type="pil", label="Detection Result")
                    upload_analysis = gr.Markdown()

            upload_btn.click(
                fn=analyze_upload,
                inputs=[upload_input, use_random_sample],
                outputs=[upload_output, upload_analysis],
            )

        if ENABLE_LIVE_STREAMING:
            with gr.Tab("Live Kamera"):
                with gr.Row():
                    with gr.Column():
                        live_input = gr.Image(
                            type="pil",
                            sources="webcam",
                            streaming=True,
                            label="Live Camera",
                        )
                    with gr.Column():
                        live_output = gr.Image(type="pil", label="Detection Result")

                live_input.stream(
                    fn=live_stream,
                    inputs=live_input,
                    outputs=live_output,
                    time_limit=30,
                    stream_every=0.5,
                )

    demo.launch(server_name="0.0.0.0")


if __name__ == "__main__":
    run()
