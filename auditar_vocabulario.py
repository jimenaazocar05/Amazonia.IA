import json
import os
from datasets import load_from_disk
from collections import Counter

# 1. Configuración
PATH_DATASET = "./mms_finetuning_data"
OUTPUT_VOCAB = "./vocab.json"

def create_vocabulary():
    print(f"📂 Cargando dataset desde: {PATH_DATASET}")
    
    try:
        dataset = load_from_disk(PATH_DATASET)
    except Exception as e:
        print(f"❌ Error cargando el dataset: {e}")
        return

    # Vamos a recolectar todos los caracteres de todos los splits (train y test)
    all_text = ""
    
    print("🔍 Analizando caracteres en el dataset...")
    
    # Unimos todo el texto disponible
    for split in dataset.keys():
        print(f"   - Procesando split: {split} ({len(dataset[split])} muestras)")
        all_text += " ".join(dataset[split]["sentence"])

    # 2. Análisis de Frecuencia
    # Esto nos sirve para ver si hay caracteres "basura" (ej: símbolos raros por error)
    caracteres_unicos = list(set(all_text))
    contador = Counter(all_text)
    
    vocab_dict = {v: k for k, v in enumerate(sorted(caracteres_unicos))}

    # 3. Reglas Especiales para Modelos de Voz (CTC)
    # El modelo necesita tokens especiales para funcionar
    
    # El espacio en blanco suele reemplazarse por una barra | para que sea visible
    if " " in vocab_dict:
        vocab_dict["|"] = vocab_dict.pop(" ")
    
    # Token de relleno (Padding) y Desconocido (Unknown)
    # Nota: Los índices pueden variar según la implementación, pero este es un estándar base
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    # 4. Guardar vocab.json
    with open(OUTPUT_VOCAB, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False, indent=2)

    print("\n" + "="*40)
    print("✅ VOCABULARIO GENERADO EXITOSAMENTE")
    print("="*40)
    print(f"Total de caracteres únicos: {len(vocab_dict)}")
    print(f"Archivo guardado en: {OUTPUT_VOCAB}")
    print("\n Auditoría de caracteres encontrados (Carácter : Frecuencia):")
    
    # Ordenar para mostrar
    for char, count in sorted(contador.items()):
        # Mostramos el espacio como [ESPACIO] para que lo veas
        char_display = "[ESPACIO]" if char == " " else char
        print(f"   '{char_display}': {count}")

    print("\n⚠️  ADVERTENCIA:")
    print("Revisa la lista de arriba. Si ves caracteres raros (ej: '$', '%', números si no corresponden),")
    print("debes limpiar tu dataset original y volver a generar la vista minable.")

if __name__ == "__main__":
    create_vocabulary();