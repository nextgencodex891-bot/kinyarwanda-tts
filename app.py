import os
from datasets import load_dataset
from TTS.utils.synthesizer import Synthesizer
import soundfile as sf
import gradio as gr

MODEL_DIR = "kinyarwanda-tts-model"

config_path = os.path.join(MODEL_DIR, "config.json")
model_path = os.path.join(MODEL_DIR, "model.pth")
se_checkpoint_path = os.path.join(MODEL_DIR, "SE_checkpoint.pth.tar")
se_config_path = os.path.join(MODEL_DIR, "config_se.json")

# Conditioning audio from dataset repo
dataset = load_dataset("nextgencodex1/kinyarwanda-conditioning-audio")
conditioning_audio_path = dataset["train"][0]["file"]

synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    encoder_checkpoint=se_checkpoint_path,
    encoder_config=se_config_path,
    use_cuda=False
)

def generate_speech(text):
    if not text:
        return None
    wav = synthesizer.tts(text, speaker_wav=conditioning_audio_path)
    output_path = "output.wav"
    sf.write(output_path, wav, 22050)
    return output_path

demo = gr.Interface(
    fn=generate_speech,
    inputs=gr.Textbox(label="Kinyarwanda Text"),
    outputs=gr.Audio(label="Generated Speech", type="filepath"),
    title="🗣️ Kinyarwanda Text-to-Speech",
    description="Convert Kinyarwanda text to natural speech."
)

if __name__ == "__main__":
    demo.launch(share=True)
