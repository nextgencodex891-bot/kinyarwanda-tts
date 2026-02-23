import os
import tempfile
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Audio
from TTS.utils.synthesizer import Synthesizer
import soundfile as sf
import gradio as gr

# =========================
# 1. Download Model Files
# =========================

REPO_ID = "nextgencodex1/kinyarwanda-tts-model"
MODEL_DIR = "kinyarwanda-tts-model"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Downloading model files...")

config_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="config.json",
    local_dir=MODEL_DIR
)

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="model.pth",
    local_dir=MODEL_DIR
)

se_checkpoint_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="SE_checkpoint.pth.tar",
    local_dir=MODEL_DIR
)

se_config_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="config_se.json",
    local_dir=MODEL_DIR
)

print("Model downloaded successfully.")

# =========================
# 2. Load Conditioning Audio Correctly
# =========================

print("Loading conditioning dataset...")

dataset = load_dataset("nextgencodex1/kinyarwanda-conditioning-audio")

# Ensure audio column is decoded
dataset = dataset.cast_column("audio", Audio())

sample = dataset["train"][0]["audio"]

audio_array = sample["array"]
sample_rate = sample["sampling_rate"]

# Save conditioning audio to temporary file
temp_audio_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
conditioning_audio_path = temp_audio_file.name

sf.write(conditioning_audio_path, audio_array, sample_rate)

print("Conditioning audio ready.")

# =========================
# 3. Initialize Synthesizer
# =========================

print("Loading TTS model...")

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    encoder_checkpoint=se_checkpoint_path,
    encoder_config=se_config_path,
    use_cuda=False  # Set True if you have GPU
)

print("Model ready.")

# =========================
# 4. Speech Generation Function
# =========================

def generate_speech(text):
    if not text.strip():
        return None

    wav = synthesizer.tts(
        text=text,
        speaker_wav=conditioning_audio_path
    )

    output_path = "output.wav"
    sf.write(output_path, wav, 22050)

    return output_path

# =========================
# 5. Gradio Interface
# =========================

demo = gr.Interface(
    fn=generate_speech,
    inputs=gr.Textbox(label="Kinyarwanda Text"),
    outputs=gr.Audio(label="Generated Speech", type="filepath"),
    title="🗣️ Kinyarwanda Text-to-Speech",
    description="Convert Kinyarwanda text to natural speech."
)

if __name__ == "__main__":
    demo.launch(share=True)
