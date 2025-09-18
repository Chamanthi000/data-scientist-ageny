import gradio as gr
import pandas as pd
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000/predict"  # adjust if serving elsewhere

def run_pipeline_ui(csv_file, target):
    import subprocess, sys, os
    # Save uploaded file to a temp path
    tmp = Path('examples/data/_upload.csv')
    if csv_file is None:
        return "Please upload a CSV."
    df = pd.read_csv(csv_file.name)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tmp, index=False)
    # Run pipeline
    cmd = [sys.executable, "-m", "src.orchestration.run_pipeline", "--data", str(tmp), "--target", target]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        return out
    except subprocess.CalledProcessError as e:
        return e.output

def predict_ui(csv_file):
    if csv_file is None:
        return {"error": 1.0}
    files = {"file": (csv_file.name, open(csv_file.name, "rb"), "text/csv")}
    try:
        r = requests.post(API_URL, files=files, timeout=30)
        return r.json()
    except Exception as e:
        return {"exception": str(e)}

with gr.Blocks() as demo:
    gr.Markdown("# Agentic Data Scientist UI")
    with gr.Tab("Train & Report"):
        csv = gr.File(label="Training CSV")
        target = gr.Textbox(label="Target Column (binary 0/1)", value="churned")
        out = gr.Textbox(label="Pipeline Logs", lines=12)
        btn = gr.Button("Run Pipeline")
        btn.click(run_pipeline_ui, inputs=[csv, target], outputs=[out])

    with gr.Tab("Predict"):
        csv2 = gr.File(label="CSV for Prediction")
        pred = gr.JSON(label="Prediction Probabilities")
        btn2 = gr.Button("Predict (calls FastAPI)")
        btn2.click(predict_ui, inputs=[csv2], outputs=[pred])

if __name__ == "__main__":
    demo.queue().launch()
