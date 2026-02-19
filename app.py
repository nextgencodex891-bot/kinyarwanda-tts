import gradio as gr
import soundfile as sf
from TTS.utils.synthesizer import Synthesizer
from normalize_rw_numbers import normalize_kinyarwanda_numbers
import os

# Permanent model folder
MODEL_DIR = r"D:\myThingAI\kinyarwanda-tts\models"

# Explicit paths
config_path = os.path.join(MODEL_DIR, "config.json")
model_path = os.path.join(MODEL_DIR, "model.pth")
se_checkpoint_path = os.path.join(MODEL_DIR, "SE_checkpoint.pth.tar")
se_config_path = os.path.join(MODEL_DIR, "config_se.json")
conditioning_audio_path = os.path.join(MODEL_DIR, "conditioning_audio.wav")

print("⚙️ Loading model...")
synthesizer = Synthesizer(
    tts_checkpoint=model_path,
    tts_config_path=config_path,
    encoder_checkpoint=se_checkpoint_path,
    encoder_config=se_config_path,
    use_cuda=False
)
print("✅ Model loaded successfully!")

def generate_speech(text):
    if not text:
        return None
    
    # Normalize numbers in text to Kinyarwanda words
    text = normalize_kinyarwanda_numbers(text)
    print(f"🎙️ Generating speech for: {text}")
    try:
        # Use conditioning_audio.wav as speaker reference to compute d-vectors
        wav = synthesizer.tts(text, speaker_wav=conditioning_audio_path)
        
        output_path = "output.wav"
        sf.write(output_path, wav, 22050)
        print("✅ Speech generated successfully!")
        
        return output_path
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

# Create Gradio interface
demo = gr.Interface(
    fn=generate_speech,
    inputs=gr.Textbox(
        label="Kinyarwanda Text", 
        placeholder="Andika inyandiko yawe hano...",
        lines=3
    ),
    outputs=gr.Audio(label="Generated Speech", type="filepath"),
    title="🎙️ Kinyarwanda Text-to-Speech",
    description="Convert Kinyarwanda text to natural speech. Running on CPU.",
    examples=[
        ["Muraho, amakuru?"],
        ["Mwaramutse neza"],
        ["Ndashimira cyane"],
        ["Murakoze"]
    ]
)

if __name__ == "__main__":
    # Use share=True if localhost is blocked
    demo.launch(share=True)