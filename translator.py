import csv
import os
import difflib
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load env to get token
load_dotenv()

class Translator:
    def __init__(self, csv_path="corpus_guahibo.csv"):
        self.csv_path = csv_path
        self.corpus = {}
        self.load_corpus()
        
        # Initialize HF Client
        # Force load .env from current directory
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        load_dotenv(env_path)
        
        token = os.getenv("HF_TOKEN")
        if not token:
            print("⚠️ Advertencia: No se encontró HF_TOKEN en .env. La traducción podría fallar.")
        
        self.client = InferenceClient(token=token)
        self.model_id = "google/madlad400-3b-mt"

    def load_corpus(self):
        """Loads the Guahibo-Spanish corpus into memory."""
        # Use absolute path for csv if not found
        if not os.path.exists(self.csv_path):
             self.csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.csv_path)

        print(f"📂 Cargando corpus desde: {self.csv_path}")
        try:
            with open(self.csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                first_row = True
                for row in reader:
                    if first_row:
                        first_row = False
                        continue
                    if len(row) >= 2:
                        es, guh = row[0].strip(), row[1].strip()
                        self.corpus[guh.lower()] = es
            print(f"✅ Corpus cargado: {len(self.corpus)} entradas.")
        except Exception as e:
            print(f"❌ Error cargando corpus: {e}")

    def find_best_local_match(self, text):
        """Busca la mejor coincidencia local usando difflib."""
        text_lower = text.lower().strip()
        
        # 1. Exact match
        if text_lower in self.corpus:
            return self.corpus[text_lower], True 

        # 2. Fuzzy match
        matches = difflib.get_close_matches(text_lower, self.corpus.keys(), n=1, cutoff=0.8)
        if matches:
            best_match = matches[0]
            print(f"💡 Coincidencia difusa: '{text}' -> '{best_match}'")
            return self.corpus[best_match], True

        return None, False

    def translate(self, text, source_lang="guh"):
        """Main translation function."""
        if not text:
            return "..."

        print(f"revisando traducción para: '{text}'")

        # 1. Check Local Corpus
        if source_lang == "guh":
            local_translation, found = self.find_best_local_match(text)
            if found:
                print("✅ Encontrado en CSV local.")
                return local_translation

        # 2. Call HF Inference API
        try:
            # MADLAD format: <2xx> Text
            # gl = Galician, es = Spanish. 
            # We need to find the code for Spanish. Standard assumes 'es'.
            # MADLAD uses specific prompt tokens. <2es> for Spanish target.
            
            prompt = f"<2es> {text}"
            print(f"🌐 Consultando HF API ({self.model_id})...")
            
            # Change to 'translation' task logic or raw post
            # The error says: Supported tasks: 'translation', got: 'text-generation'
            # We use translation() helper or post()
            
            response = self.client.translation(
                text=prompt,
                model=self.model_id
            )
            
            # Response from translation task is usually { 'translation_text': '...' } or list of dicts at root
            if isinstance(response, dict) and 'translation_text' in response:
                 return response['translation_text']
            elif isinstance(response, list) and len(response) > 0 and 'translation_text' in response[0]:
                 return response[0]['translation_text']
            else:
                 # If returns raw string
                 return str(response).strip()

        except Exception as e:
            print(f"❌ Error API HF: {e}")
            return f"Error de traducción: {str(e)}"

# Singleton instance for simple import
translator_instance = Translator()
