import os
import shutil
from datasets import load_from_disk

# --- CONFIGURACIÓN ---
PATH_ORIGINAL = "./mms_finetuning_data"
PATH_TEMP     = "./mms_finetuning_data_temp"  # Carpeta temporal

# --- DICCIONARIO DE LIMPIEZA ---
reemplazos = {
    "ί": "i",
    "α": "a",
    "ρ": "p", 
    "ế": "e",
    "c": "k",
    # Puedes agregar más aquí si ves más basura en el futuro
}

def limpiar_texto(batch):
    texto_original = batch["sentence"]
    texto_nuevo = texto_original
    
    # Aplicar reemplazos
    for sucio, limpio in reemplazos.items():
        texto_nuevo = texto_nuevo.replace(sucio, limpio)
    
    batch["sentence"] = texto_nuevo
    return batch

def main():
    print(f"🧹 Cargando dataset desde {PATH_ORIGINAL}...")
    
    try:
        dataset = load_from_disk(PATH_ORIGINAL)
    except FileNotFoundError:
        print("❌ Error: No se encuentra la carpeta del dataset.")
        return

    print("🧼 Limpiando caracteres extraños...")
    dataset_limpio = dataset.map(limpiar_texto)

    # 1. Guardar en una carpeta NUEVA (Temporal) para evitar el error de Permisos
    print(f"💾 Guardando versión limpia en ruta temporal: {PATH_TEMP}...")
    dataset_limpio.save_to_disk(PATH_TEMP)
    print("✅ Guardado temporal exitoso.")

    # 2. Intentar reemplazar la carpeta original
    print("🔄 Sustituyendo carpeta original por la limpia...")
    try:
        # En Windows a veces los archivos quedan "tomados" por Python un momento.
        # Intentamos borrar la vieja y renombrar la nueva.
        shutil.rmtree(PATH_ORIGINAL)
        os.rename(PATH_TEMP, PATH_ORIGINAL)
        print("✅ ¡TRANSFORMACIÓN COMPLETADA! La carpeta 'mms_finetuning_data' ahora está limpia.")
    except PermissionError:
        print("\n⚠️  ALERTA DE WINDOWS: No se pudo borrar la carpeta original automáticamente.")
        print(f"No te preocupes. Tu dataset limpio está guardado en: {PATH_TEMP}")
        print("Paso manual: Borra la carpeta 'mms_finetuning_data' y renombra la carpeta '_temp' a ese nombre.")
    except Exception as e:
        print(f"❌ Ocurrió un error moviendo carpetas: {e}")

if __name__ == "__main__":
    main()