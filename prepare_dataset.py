import os
import glob
import pandas as pd
from datasets import Dataset, Audio, DatasetDict

# --- CONFIGURACIÓN ---
BASE_DIR = "./datos_finales"
PATH_CLEAN_TRAIN = os.path.join(BASE_DIR, "trainClean_wav")
PATH_AUG_TRAIN   = os.path.join(BASE_DIR, "trainAugmentation_wav")
PATH_TEST        = os.path.join(BASE_DIR, "test_wav")

SAMPLING_RATE = 16000

def normalize_id(path_str):
    """
    Convierte rutas sucias como 'trainClean_wav\audio_59_clean.wav'
    a un ID limpio: 'audio_59_clean'
    """
    # Eliminar extensión
    name = os.path.splitext(os.path.basename(path_str))[0]
    return name.strip()

def load_data_from_folder(folder_path, category_label, csv_path):
    data = []
    
    # 1. Cargar CSV y crear diccionario de búsqueda
    df = pd.read_csv(csv_path)
    text_mapping = {}
    
    print(f"   - Indexando CSV {os.path.basename(csv_path)}...")
    for _, row in df.iterrows():
        # Asumiendo columnas 'audio_path' y 'sentence' que vi en tu archivo
        # Limpiamos el ID del CSV (quitamos carpetas y barras)
        raw_id = str(row['audio_path'])
        clean_id = normalize_id(raw_id)
        
        text = str(row['sentence']).strip()
        text_mapping[clean_id] = text

    # 2. Buscar archivos físicos
    files = glob.glob(os.path.join(folder_path, "*.wav"))
    print(f"   - Procesando {len(files)} audios en {folder_path}...")
    
    found_count = 0
    missing_count = 0
    
    for file_path in files:
        # Obtenemos el ID del archivo físico actual
        file_name = os.path.basename(file_path)
        file_id = normalize_id(file_name)
        
        transcription = None
        
        # CASO 1: Búsqueda directa (ej: audio_59_clean)
        if file_id in text_mapping:
            transcription = text_mapping[file_id]
            
        # CASO 2: Manejo de Aumentación (ej: audio_59_aug_0 -> buscar audio_59_clean)
        # Tu CSV tiene las rutas de aumentación, pero a veces los nombres varían.
        # Si no lo encontró directo, probamos lógica de limpieza.
        if not transcription and "_aug_" in file_id:
            # Intentar reconstruir el ID original. 
            # Suponiendo que audio_59_aug_0 viene de audio_59_clean
            # Estrategia: buscar la raíz "audio_59"
            parts = file_id.split("_aug_")
            base_part = parts[0] # audio_59
            
            # Buscamos en el mapa algo que empiece igual
            # (Esto es lento pero seguro para datasets pequeños)
            for k, v in text_mapping.items():
                if k.startswith(base_part + "_clean") or k == base_part:
                    transcription = v
                    break

        if transcription:
            data.append({
                "audio": file_path,
                "sentence": transcription,
                "category": category_label
            })
            found_count += 1
        else:
            # 🚨 ERROR: No agregar si no hay texto. Mejor perder un audio que dañar el modelo.
            # print(f"❌ Texto no encontrado para: {file_id}") 
            missing_count += 1

    print(f"   ✅ Encontrados: {found_count} | ❌ Perdidos (sin texto): {missing_count}")
    return data

# --- EJECUCIÓN ---
print("🚀 Iniciando regeneración de Vista Minable (V2 - Strict)...")

csv_train_path = os.path.join(BASE_DIR, "train.csv")
csv_test_path  = os.path.join(BASE_DIR, "test.csv")

# 1. Cargar datos
print("\n📂 Cargando Train Clean...")
ds_clean = load_data_from_folder(PATH_CLEAN_TRAIN, "clean", csv_train_path)

print("\n📂 Cargando Train Augmentation...")
ds_aug = load_data_from_folder(PATH_AUG_TRAIN, "augmented", csv_train_path)

print("\n📂 Cargando Test...")
ds_test_list = load_data_from_folder(PATH_TEST, "test", csv_test_path)

# 2. Unificar
full_train_list = ds_clean + ds_aug

print(f"\n📊 Resumen Final:")
print(f"   - Train Total: {len(full_train_list)}")
print(f"   - Test Total: {len(ds_test_list)}")

if len(full_train_list) == 0:
    raise ValueError("⚠️  FATAL: No se cargaron datos de entrenamiento. Revisa los nombres en el CSV.")

# 3. Crear DatasetDict
vista_minable = DatasetDict({
    "train": Dataset.from_list(full_train_list),
    "test_synthetic": Dataset.from_list(ds_test_list)
})

# 4. Casting
print("🔊 Normalizando audio a 16kHz...")
vista_minable = vista_minable.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))

# 5. Guardar
output_path = "./mms_finetuning_data"
vista_minable.save_to_disk(output_path)