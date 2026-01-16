import os
import csv
import librosa
import soundfile as sf
import datetime

# Rutas de guardado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "datos_finales")
WAV_DIR = os.path.join(DATA_DIR, "nuevos_audios_wav")
CSV_PATH = os.path.join(DATA_DIR, "nuevos_datos.csv")

def ensure_directories():
    """Asegura que existan los directorios y archivos necesarios."""
    if not os.path.exists(WAV_DIR):
        os.makedirs(WAV_DIR)
        print(f"Directorio creado: {WAV_DIR}")
    
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["audio_path", "sentence"])
        print(f"CSV creado: {CSV_PATH}")

def save_contribution(audio_path, text):
    """
    Procesa y guarda el audio grabado y el texto asociado.
    
    Args:
        audio_path (str): Ruta temporal del archivo de audio grabado por Gradio.
        text (str): Texto transcrito por el usuario (Ground Truth).
        
    Returns:
        str: Mensaje de éxito o error.
    """
    if audio_path is None:
        return "⚠️ Error: No se ha detectado ningún audio cargado o grabado."
    
    if not text or not text.strip():
        return "⚠️ Error: El campo de texto está vacío. Por favor, escribe la transcripción."

    try:
        ensure_directories()
        
        # 1. Generar nombre de archivo único con timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rec_{timestamp}.wav"
        save_path = os.path.join(WAV_DIR, filename)
        
        # 2. Cargar y Procesar Audio (Resample a 16kHz Mono)
        # librosa carga a mono=True por defecto y hace resample si se especifica sr
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 3. Guardar Audio Procesado (Temporalmente en disco)
        sf.write(save_path, y, 16000)
        
        # 4. SUBIDA A LA NUBE (Hugging Face)
        from cloud_handler import cloud_handler
        
        msg_extra = ""
        if cloud_handler:
            success, msg = cloud_handler.upload_audio(save_path, text.strip())
            if success:
                msg_extra = f"\n☁️ {msg}"
                # Opcional: Borrar archivo local si ya está seguro en la nube
                # os.remove(save_path) 
            else:
                msg_extra = f"\n⚠️ Falló subida a nube: {msg}"
        else:
            msg_extra = "\n⚠️ Nube no configurada (.env faltante)"

        # 5. Guardar en CSV Local (Registro de respaldo)
        # Usamos ruta relativa para el CSV: nuevos_audios_wav/filename.wav
        relative_path = os.path.join("nuevos_audios_wav", filename)
        
        with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([relative_path, text.strip()])
            
        return f"✅ ¡Guardado Exitoso! \nAudio: {filename}\nTexto: {text}{msg_extra}"
        
    except Exception as e:
        print(f"Error al guardar contribución: {e}")
        return f"❌ Error interno al guardar: {str(e)}"
