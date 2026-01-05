import gradio as gr
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings
import os

# Configuración
warnings.filterwarnings("ignore")
MODELO_PATH = "./modelo_guahibo_gpu"

print("⏳ Cargando modelo para la App... Espere un momento.")

# 1. Cargar Modelo (Se carga una sola vez al inicio)
try:
    processor = Wav2Vec2Processor.from_pretrained(MODELO_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODELO_PATH)
    
    # Usar GPU si hay, si no CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Modelo cargado en: {device.upper()}")
except Exception as e:
    print(f"❌ Error fatal cargando modelo: {e}")
    exit()

# 2. Función de Transcripción (La lógica que ya conoces)
def transcribir(audio_path):
    if audio_path is None:
        return "⚠️ Por favor, graba o sube un audio."
    
    try:
        # Cargar audio con Librosa (fuerza 16kHz)
        audio, rate = librosa.load(audio_path, sr=16000)
        
        # Procesar
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        
        # Predicción
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        
        # Decodificar
        pred_ids = torch.argmax(logits, dim=-1)
        texto = processor.batch_decode(pred_ids)[0]
        
        return texto
    except Exception as e:
        return f"Error procesando: {e}"

# 3. Interfaz Gráfica (El Frontend)
# Diseñamos la web aquí mismo
with gr.Blocks(title="Amazonia.IA") as demo:
    gr.Markdown(
        """
        # 🦜 Amazonia.IA: Traductor Jivi-Guahibo
        **Sistema de Reconocimiento Automático de Habla (ASR) para lenguas indígenas.**
        *Desarrollado como prototipo académico.*
        """
    )
    
    with gr.Row():
        with gr.Column():
            # Entrada: Micrófono o Archivo
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Habla o Sube un Audio")
            btn_transcribir = gr.Button("Transcribir Audio", variant="primary")
        
        with gr.Column():
            # Salida: Texto
            text_output = gr.Textbox(label="📝 Transcripción generada por la IA:", lines=4)
    
    # Conectar el botón con la función
    btn_transcribir.click(fn=transcribir, inputs=audio_input, outputs=text_output)

    gr.Markdown("---")
    gr.Markdown("By Jimena Azocar, Joaquim Juha, Carlos Mendez, Stalin Franco - UCAB Guayana")

# 4. Lanzar la App
if __name__ == "__main__":
    demo.launch(inbrowser=True)