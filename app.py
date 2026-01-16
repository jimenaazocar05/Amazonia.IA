import gradio as gr
import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings
import os

# --- CONFIGURACIÓN ---
warnings.filterwarnings("ignore")
MODELO_PATH = "./modelo_guahibo_gpu"

# URL de la imagen de fondo (Local) - Usando Base64 para evitar errores de ruta
import os
import base64

def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f"data:image/webp;base64,{encoded_string}"
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return ""

# Ruta relativa simple ya que assets está en el root
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "assets", "imagenFondo.webp")
B64_FONDO = encode_image(IMAGE_PATH)

# Importar traductor
from translator import translator_instance
from data_handler import save_contribution

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
    background-image: url('{B64_FONDO}') !important;
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
        
        # --- TRADUCCIÓN ---
        traduccion = translator_instance.translate(texto)
        
        return texto, traduccion
    except Exception as e:
        return f"Error: {e}", "Error procesando el audio"

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
    
    with gr.Tabs():
        # --- TAB 1: TRADUCTOR (Existente) ---
        with gr.TabItem("Traductor"):
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
                    text_output = gr.Textbox(
                        label="Transcripción (Jivi)", 
                        lines=3,
                        placeholder="El texto en guahibo aparecerá aquí..."
                    )
                    translation_output = gr.Textbox(
                        label="Traducción (Español)", 
                        lines=3,
                        placeholder="La traducción aparecerá aquí..."
                    )
            
            btn_transcribir.click(fn=transcribir, inputs=audio_input, outputs=[text_output, translation_output])

        # --- TAB 2: CONTRIBUIR (Nuevo) ---
        with gr.TabItem("Contribuir / Entrenar"):
            gr.Markdown("### 🎙️ Ayuda a mejorar el modelo")
            gr.Markdown("Graba tu voz y escribe exactamente lo que dijiste en Guahibo. Estos datos se usarán para reentrenar la IA.")
            
            with gr.Row():
                with gr.Column():
                    # Input exclusivo de micrófono para nuevos datos
                    rec_input = gr.Audio(
                        sources=["microphone"], 
                        type="filepath", 
                        label="Grabar Voz (Solo Micrófono)",
                        interactive=True
                    )
                with gr.Column():
                    transcription_input = gr.Textbox(
                        label="Texto en Guahibo (Lo que dijiste)", 
                        placeholder="Escribe aquí la transcripción exacta...",
                        lines=5
                    )
            
            btn_save = gr.Button("💾 Guardar Contribución", variant="secondary", size="lg")
            status_output = gr.Textbox(label="Estado", interactive=False)
            
            btn_save.click(
                fn=save_contribution,
                inputs=[rec_input, transcription_input],
                outputs=status_output
            )

    gr.Markdown("---")
    with gr.Column(scale=1, min_width=200):
         gr.Markdown("By Jimena Azocar, Joaquim Juha, Carlos Mendez, Stalin Franco - UCAB Guayana", elem_classes=["text-right"])
         gr.Markdown("**Integración:** Modelo Wav2Vec2 (Local) + API Traductor (Remoto) + Corpus CSV", elem_classes=["text-right"])

# Lanzar
if __name__ == "__main__":
    demo.launch(inbrowser=True, allowed_paths=["/tmp/"])