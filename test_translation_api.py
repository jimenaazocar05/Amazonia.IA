from translator import translator_instance
import time
import traceback

def test():
    print("--- Test Traductor ---")
    
    test_phrase = "Jajapawanaxaneto" 
    print(f"\nProbando traducción de: '{test_phrase}' (Sikuani -> Español)")
    
    start = time.time()
    try:
        res = translator_instance.translate(test_phrase)
        print(f"Resultado: {res}")
    except Exception:
        traceback.print_exc()
        
    end = time.time()
    print(f"Tiempo: {end - start:.2f}s")

if __name__ == "__main__":
    test()
