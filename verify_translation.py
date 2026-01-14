from translator import Translator
import os

def test_translation():
    print("🧪 Iniciando pruebas de traducción...")
    
    # 1. Initialize
    t = Translator()
    
    # 2. Test Local Cache (Exact Match)
    print("\n[Prueba 1] Búsqueda Local Exacta")
    # 'Necobeyo. Nawiatajë rabaja.' -> 'Su mano. Ya me voy (regreso a mi casa).' 
    # But keys are lowercased and stripped.
    # CSV Line 2: Su mano. Ya me voy (regreso a mi casa).,Necobeyo. Nawiatajë rabaja.
    guahibo_phrase = "Necobeyo. Nawiatajë rabaja."
    translation = t.translate(guahibo_phrase)
    print(f"Input: {guahibo_phrase}")
    print(f"Output: {translation}")
    
    if "Su mano" in translation:
        print("✅ Prueba 1 Pasada: Traducción local encontrada.")
    else:
        print("❌ Prueba 1 Fallada.")

    # 3. Test Fuzzy Match
    print("\n[Prueba 2] Búsqueda Difusa")
    # Small typo: "Necobeyo" -> "Necobeyo."
    fuzzy_input = "Necobeyo" 
    # CSV has "Necobeyo. Nawiatajë rabaja." but maybe shorter phrases exist.
    # Line 2: Necobeyo. Nawiatajë rabaja.
    # Line 521: Necobeyo.
    translation_fuzzy = t.translate(fuzzy_input)
    print(f"Input: {fuzzy_input}")
    print(f"Output: {translation_fuzzy}")
    if "Su mano" in translation_fuzzy:
         print("✅ Prueba 2 Pasada: Coincidencia difusa encontrada.")
    else:
         print("⚠️ Prueba 2: No se encontró coincidencia difusa (comportamiento aceptable si no hay cercana).")

    # 4. Test Quota File Creation
    print("\n[Prueba 3] Archivo de Cuota")
    if os.path.exists(".daily_quota.json"):
        print("✅ Archivo .daily_quota.json existe.")
    else:
        print("⚠️ Archivo .daily_quota.json NO existe (quizás no se hizo llamada API aún).")

    print("\n🏁 Pruebas finalizadas.")

if __name__ == "__main__":
    test_translation()
