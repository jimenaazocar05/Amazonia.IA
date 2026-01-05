import gradio as gr
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings
import os

# --- CONFIGURACIÓN ---
warnings.filterwarnings("ignore")
MODELO_PATH = "./modelo_guahibo_gpu"

# URL de la imagen de fondo (Selva Amazónica)
URL_FONDO_AMAZONIA = "https://www.mashpilodge.com/wp-content/uploads/2025/07/AdobeStock_1042605634-2048x1148.jpeg"

# --- 1. PERSONALIZACIÓN VISUAL ---

# A) Definir el Tema (AHORA NARANJA)
tema_amazonia = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange, 
    neutral_hue=gr.themes.colors.slate,    
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    button_primary_text_color="white",
)

# B) Definir CSS
css_personalizado = f"""
.gradio-container {{
    background-image: url('{URL_FONDO_AMAZONIA}') !important;
    background-size: cover !important;
    background-position: center !important;
    background-attachment: fixed !important;
    background-repeat: no-repeat !important;
}}

/* Contenedor semitransparente */
.gradio-container > .main {{
    background-color: rgba(255, 255, 255, 0.90) !important; 
    backdrop-filter: blur(5px);
    border-radius: 20px;
    padding: 30px !important;
    margin-top: 20px !important;
    box-shadow: 0 8px 32px 0 rgba( 31, 38, 135, 0.37 );
    max-width: 1200px;
    margin-left: auto;
    margin-right: auto;
}}

.dark .gradio-container > .main {{
     background-color: rgba(0, 0, 0, 0.75) !important;
}}

/* Título Principal */
#titulo-principal h1 {{
    color: #c2410c;  
    font-weight: 900;
    font-size: 2.5em;
    text-align: center;
}}
"""

print("⏳ Cargando modelo para la App... Espere un momento.")

# --- 2. CARGA DEL MODELO ---
try:
    processor = Wav2Vec2Processor.from_pretrained(MODELO_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODELO_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"✅ Modelo cargado en: {device.upper()}")
except Exception as e:
    print(f"❌ Error fatal cargando modelo: {e}")
    exit()

# --- 3. FUNCIÓN DE TRANSCRIPCIÓN ---
def transcribir(audio_path):
    if audio_path is None:
        return "⚠️ Por favor, graba o sube un audio antes de transcribir."
    
    print(f"🎤 Procesando audio: {audio_path}...")
    try:
        # Cargar con librosa
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
        return f"Error procesando el audio. Detalles: {e}"

# --- 4. INTERFAZ GRÁFICA ---
with gr.Blocks(title="Amazonia.IA", theme=tema_amazonia, css=css_personalizado) as demo:
    
    with gr.Column(elem_id="titulo-principal"):
        gr.Markdown(
            """
            # 🦜 Amazonia.IA
            ### Traductor Automático Jivi (Guahibo) - Español
            """
        )
        gr.Markdown("Proyecto de prototipo académico para la preservación de lenguas indígenas.")
        gr.Markdown("---")
    
    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="Graba o Sube tu audio",
                interactive=True
            )
            btn_transcribir = gr.Button("Transcribir Audio", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            # Aquí quité el 'show_copy_button' que daba error
            text_output = gr.Textbox(
                label="Resultado de la IA", 
                lines=6,
                placeholder="La transcripción aparecerá aquí..."
            )
    
    btn_transcribir.click(fn=transcribir, inputs=audio_input, outputs=text_output)

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
             gr.Markdown("By Jimena Azocar, Joaquim Juha, Carlos Mendez, Stalin Franco - UCAB Guayana", elem_classes=["text-right"])

# Lanzar
if __name__ == "__main__":
    demo.launch(inbrowser=True, allowed_paths=["/tmp/"])