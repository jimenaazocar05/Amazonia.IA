import csv
import json
import os
import requests
import datetime
import difflib
from urllib.parse import quote

class Translator:
    def __init__(self, csv_path="corpus_guahibo.csv", quota_file=".daily_quota.json", api_url="https://traductor-guahibo.cca-gomez2014.workers.dev/"):
        self.csv_path = csv_path
        self.quota_file = quota_file
        self.api_url = api_url
        self.max_daily_requests = 20
        self.corpus = {}
        self.load_corpus()

    def load_corpus(self):
        """Loads the Guahibo-Spanish corpus into memory."""
        print(f"📂 Cargando corpus desde: {self.csv_path}")
        try:
            with open(self.csv_path, mode='r', encoding='utf-8') as f:
                reader = csv.reader(f)
                # Assuming format: spanish, guahibo based on file view
                # Line 1: es,guh
                first_row = True
                for row in reader:
                    if first_row:
                        first_row = False
                        continue
                    if len(row) >= 2:
                        es, guh = row[0].strip(), row[1].strip()
                        # Normalize keys for better matching
                        self.corpus[guh.lower()] = es
            print(f"✅ Corpus cargado: {len(self.corpus)} entradas.")
        except Exception as e:
            print(f"❌ Error cargando corpus: {e}")

    def check_and_update_quota(self):
        """Checks if we have API credits left for today."""
        today = datetime.date.today().isoformat()
        
        data = {"date": today, "count": 0}
        
        if os.path.exists(self.quota_file):
            try:
                with open(self.quota_file, 'r') as f:
                    file_data = json.load(f)
                    if file_data.get("date") == today:
                        data = file_data
            except Exception:
                pass # Corrupt file, reset

        if data["count"] >= self.max_daily_requests:
            print("⚠️ Límite diario de API alcanzado.")
            return False
        
        # Increment and save (optimistic: assuming we will make the call)
        data["count"] += 1
        with open(self.quota_file, 'w') as f:
            json.dump(data, f)
        
        print(f"📊 Uso de API hoy: {data['count']}/{self.max_daily_requests}")
        return True

    def find_best_local_match(self, text):
        """Busca la mejor coincidencia local usando difflib."""
        text_lower = text.lower().strip()
        
        # 1. Exact match
        if text_lower in self.corpus:
            return self.corpus[text_lower], True # Exact match

        # 2. Fuzzy match
        # Buscamos claves del corpus
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

        # 1. Check Local Corpus (ONLY for Guahibo -> Spanish for now as per key structure)
        if source_lang == "guh":
            local_translation, found = self.find_best_local_match(text)
            if found:
                print("✅ Encontrado en CSV local.")
                return local_translation

        # 2. Check Quota
        if not self.check_and_update_quota():
            return "⚠️ Límite de traducción online alcanzado. Intenta con frases del corpus."

        # 3. Call API with Retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # "Yo quiero" -> "Yo%20quiero"
                params = {
                    "frase": text,
                    "origen": source_lang
                }
                
                print(f"🌐 Consultando API (Intento {attempt + 1}/{max_retries}): {self.api_url} ...")
                # Increased timeout to 30s to handle cold starts
                response = requests.get(self.api_url, params=params, timeout=30)
                
                if response.status_code != 200:
                    return f"Error HTTP: {response.status_code}"
                
                data = response.json()
                
                if data.get("status") == "success":
                    return data.get("traduccion")
                else:
                    return f"Error API: {data.get('mensaje')}"

            except requests.exceptions.Timeout:
                print(f"⚠️ Timeout en intento {attempt + 1}.")
                if attempt == max_retries - 1:
                    return "Error: El servidor de traducción tardó demasiado en responder."
            except Exception as e:
                return f"Error de conexión: {str(e)}"
        
        return "Error desconocido después de varios intentos."

# Singleton instance for simple import
translator_instance = Translator()
