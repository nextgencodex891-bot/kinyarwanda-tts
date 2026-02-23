import os
import tempfile
from huggingface_hub import hf_hub_download
from datasets import load_dataset, Audio
from TTS.utils.synthesizer import Synthesizer
import soundfile as sf
import gradio as gr

# ─────────────────────────────────────────
# 1. Model Download (cached after first run)
# ─────────────────────────────────────────
REPO_ID    = "nextgencodex1/kinyarwanda-tts-model"
MODEL_DIR  = "/workspace/kinyarwanda-tts-model"

os.makedirs(MODEL_DIR, exist_ok=True)

def download_if_missing(filename):
    local_path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {filename}...")
        hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir=MODEL_DIR)
    else:
        print(f"✅ {filename} already cached.")
    return local_path

config_path        = download_if_missing("config.json")
model_path         = download_if_missing("model.pth")
se_checkpoint_path = download_if_missing("SE_checkpoint.pth.tar")
se_config_path     = download_if_missing("config_se.json")

# ─────────────────────────────────────────
# 2. Load Conditioning Audio
# ─────────────────────────────────────────
CONDITIONING_CACHE = "/workspace/conditioning.wav"

if not os.path.exists(CONDITIONING_CACHE):
    print("Loading conditioning dataset...")
    dataset = load_dataset("nextgencodex1/kinyarwanda-conditioning-audio")
    dataset = dataset.cast_column("audio", Audio())
    sample      = dataset["train"][0]["audio"]
    audio_array = sample["array"]
    sample_rate = sample["sampling_rate"]
    sf.write(CONDITIONING_CACHE, audio_array, sample_rate)
    print("✅ Conditioning audio saved.")
else:
    print("✅ Conditioning audio already cached.")

conditioning_audio_path = CONDITIONING_CACHE

# ─────────────────────────────────────────
# 3. Load TTS Synthesizer
# ─────────────────────────────────────────
print("Loading TTS model...")
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    encoder_checkpoint=se_checkpoint_path,
    encoder_config=se_config_path,
    use_cuda=False  # Set True if GPU is available
)
print("✅ Model ready.")

# ─────────────────────────────────────────
# 4. Speech Generation
# ─────────────────────────────────────────
def generate_speech(text):
    if not text.strip():
        return None
    wav = synthesizer.tts(
        text=text,
        speaker_wav=conditioning_audio_path
    )
    output_path = "/workspace/output.wav"
    sf.write(output_path, wav, 22050)
    return output_path

# ─────────────────────────────────────────
# 5. Gradio Interface
# ─────────────────────────────────────────
demo = gr.Interface(
    fn=generate_speech,
    inputs=gr.Textbox(label="Kinyarwanda Text", placeholder="Andika hano..."),
    outputs=gr.Audio(label="Generated Speech", type="filepath"),
    title="🇷🇼 Kinyarwanda Text-to-Speech",
    description="Convert Kinyarwanda text to natural speech using AI.",
    examples=[["Muraho neza"], ["Amakuru?"], ["Ndagukunda cyane"]],
)

if __name__ == "__main__":
    demo.launch(share=True, server_port=7861, server_name="0.0.0.0")
