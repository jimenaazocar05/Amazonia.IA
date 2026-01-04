import torch
import librosa
import os
import glob 
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import warnings

# Limpiar la salida de la consola
os.system('cls' if os.name == 'nt' else 'clear')
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN ---
MODELO_PATH = "./modelo_guahibo_gpu"
CARPETA_AUDIOS = "./audios_wav"

print("----------------------------------------------------------------")
print("🦜 AMAZONIA.IA - MÓDULO DE TRANSCRIPCIÓN AUTOMÁTICA (GUAHIBO)")
print("----------------------------------------------------------------")

def cargar_modelo():
    print("⏳ Cargando el cerebro de la IA... (esto puede tardar unos segundos)")
    try:
        # Intentamos cargar desde la carpeta local
        processor = Wav2Vec2Processor.from_pretrained(MODELO_PATH)
        model = Wav2Vec2ForCTC.from_pretrained(MODELO_PATH)
        
        # Mover a GPU si tienes NVIDIA 
        if torch.cuda.is_available():
            model.to("cuda")
            print("🔥 GPU DETECTADA: Aceleración activada.")
        else:
            print("⚠️ GPU NO DETECTADA: Usando CPU (más lento).")
            
        return processor, model
    except Exception as e:
        print(f"\n❌ ERROR CRÍTICO: No encuentro el modelo en '{MODELO_PATH}'")
        print(f"Detalle: {e}")
        return None, None

def transcribir_archivo(ruta_archivo, processor, model):
    try:
        # 1. Cargar audio y forzar a 16kHz (Wav2Vec2 exige 16k)
        audio_input, _ = librosa.load(ruta_archivo, sr=16000)

        # 2. Convertir audio a números (Tensores)
        inputs = processor(audio_input, sampling_rate=16000, return_tensors="pt", padding=True)

        # 3. Mover datos a la misma dispositivo que el modelo (GPU o CPU)
        device = model.device
        input_values = inputs.input_values.to(device)

        # 4. Predicción (La IA escucha)
        with torch.no_grad():
            logits = model(input_values).logits

        # 5. Decodificar (Números -> Letras)
        pred_ids = torch.argmax(logits, dim=-1)
        transcripcion = processor.batch_decode(pred_ids)[0]
        
        return transcripcion

    except Exception as e:
        return f"Error procesando audio: {e}"

# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # 1. Cargar modelo
    processor, model = cargar_modelo()

    if model:
        # 2. Buscar todos los .wav en la carpeta
        archivos = glob.glob(os.path.join(CARPETA_AUDIOS, "*.wav"))
        
        if not archivos:
            print(f"\n⚠️ No encontré audios .wav en la carpeta '{CARPETA_AUDIOS}'")
        else:
            print(f"\n📂 Encontré {len(archivos)} archivo(s) para procesar.\n")

            for archivo in archivos:
                nombre_archivo = os.path.basename(archivo)
                print(f"🎧 Escuchando: {nombre_archivo} ...")
                
                texto = transcribir_archivo(archivo, processor, model)
                
                print(f"📝 Transcripción: \"{texto}\"")
                print("-" * 40)
            
            print("\n✅ Proceso finalizado.")