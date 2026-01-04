import torch
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import evaluate
import pandas as pd
import os
import sys
import librosa  
import warnings

# Ignorar advertencias de hash/cache
warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN ---
MODELO_PATH = "./modelo_guahibo_gpu"
CARPETA_TEST = "./datos_finales/test_wav" 
RESULTADOS_CSV = "resultados_custom.csv"

# Limpiar pantalla
os.system('cls' if os.name == 'nt' else 'clear')
print("📊 --- EVALUACIÓN FINAL (MODO LIBROSA) --- 📊")

# 1. Cargar Métricas
print("1️⃣ Cargando métricas...")
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

# 2. Cargar Modelo
print("2️⃣ Cargando modelo...")
try:
    processor = Wav2Vec2Processor.from_pretrained(MODELO_PATH)
    model = Wav2Vec2ForCTC.from_pretrained(MODELO_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    sys.exit()

# 3. Cargar CSV Manualmente
archivo_csv = os.path.join(CARPETA_TEST, "metadata.csv")
print(f"3️⃣ Leyendo metadatos desde: {archivo_csv} ...")

if not os.path.exists(archivo_csv):
    print("❌ ERROR: No encuentro el archivo metadata.csv.")
    sys.exit()

try:
    df = pd.read_csv(archivo_csv)
    
    # Parche para punto y coma
    if "file_name" not in df.columns and len(df.columns) == 1:
        df = pd.read_csv(archivo_csv, sep=";")
    
    df.columns = [c.strip() for c in df.columns]

    if "file_name" not in df.columns:
        print(f"❌ Error en columnas CSV. Se encontró: {df.columns.tolist()}")
        sys.exit()
        
    print(f"   ✅ Se encontraron {len(df)} registros.")

    # Construir rutas completas
    df["audio_path"] = df["file_name"].apply(lambda x: os.path.join(CARPETA_TEST, x))
    
    # Crear Dataset simple (sin cast_column para evitar torchcodec)
    dataset = Dataset.from_pandas(df)

except Exception as e:
    print(f"❌ Error leyendo el CSV: {e}")
    sys.exit()

# 4. Función de Predicción (USANDO LIBROSA)
def predecir(batch):
    try:
        ruta_archivo = batch["audio_path"]
        
        # --- CAMBIO CLAVE: Cargar con librosa ---
        # Esto evita el error de torchcodec porque librosa usa soundfile
        audio_array, _ = librosa.load(ruta_archivo, sr=16000)
        
        # Procesar audio
        inputs = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_str"] = processor.batch_decode(pred_ids)[0]
        
        # Normalizar texto real
        col_texto = "sentence" if "sentence" in batch else "text"
        batch["target_str"] = str(batch[col_texto]).lower()
        
    except Exception as e:
        print(f"⚠️ Error en archivo {ruta_archivo}: {e}")
        batch["pred_str"] = ""
        batch["target_str"] = ""
    
    return batch

# 5. Ejecutar
print("\n🚀 4️⃣ Analizando audios...")
results = dataset.map(predecir)

# 6. Calcular Métricas
print("\n🧮 5️⃣ Calculando resultados...")
predictions = [p for p in results["pred_str"] if p]
references = [r for r in results["target_str"] if r]

if len(predictions) == 0:
    print("❌ No se pudo predecir nada.")
    sys.exit()

wer = wer_metric.compute(predictions=predictions, references=references)
cer_score = cer_metric.compute(predictions=predictions, references=references)

# 7. Reporte
print("\n" + "="*40)
print(f"🌟 RESULTADOS DE TUS AUDIOS DE PRUEBA")
print("="*40)
print(f"📉 WER (Error Palabras):   {wer:.2%}")
print(f"📉 CER (Error Caracteres): {cer_score:.2%}")
print(f"✅ Precisión Fonética:     {1 - cer_score:.2%}")
print("="*40)

# Guardar Excel
df_result = pd.DataFrame({
    "Archivo": df["file_name"], 
    "Realidad": references, 
    "Predicción": predictions
})
df_result.to_csv(RESULTADOS_CSV, index=False, encoding="utf-8-sig")
print(f"\n💾 Detalle guardado en: {RESULTADOS_CSV}")