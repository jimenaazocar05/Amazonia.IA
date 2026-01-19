import os
import csv
import io
import time
from dotenv import load_dotenv
from huggingface_hub import HfApi, HfFileSystem

# Cargar variables de entorno
# Force load from current directory
env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("HF_REPO_ID")

class CloudHandler:
    def __init__(self):
        global HF_TOKEN, REPO_ID
        if not HF_TOKEN or not REPO_ID:
             # Retry loading checking env again
            if not HF_TOKEN: HF_TOKEN = os.getenv("HF_TOKEN")
            if not REPO_ID: REPO_ID = os.getenv("HF_REPO_ID")
            
            if not HF_TOKEN or not REPO_ID:
                print(f"⚠️ DEBUG: Env Path tried: {env_path}")
                print(f"⚠️ DEBUG: HF_TOKEN present: {bool(HF_TOKEN)}")
                print(f"⚠️ DEBUG: REPO_ID present: {bool(REPO_ID)}")
                raise ValueError("❌ Faltan las credenciales (HF_TOKEN o HF_REPO_ID) en el archivo .env")
        
        self.api = HfApi(token=HF_TOKEN)
        self.fs = HfFileSystem(token=HF_TOKEN)
        self.repo_id = REPO_ID
        self.dataset_type = "dataset" # Siempre 'dataset' para almacenamiento de datos
        
        # Verificar que el repo existe, si no, intentar crearlo (opcional, asume que existe)
        print(f"☁️ Conectado a Hugging Face: {self.repo_id}")

    def upload_audio(self, local_path, text, relative_folder="data"):
        """
        Sube un audio a Hugging Face y actualiza el metadata.csv remoto.
        """
        try:
            # 1. Generar nombre remoto único
            filename = os.path.basename(local_path)
            remote_path = f"{relative_folder}/{filename}"
            
            # 2. Subir el archivo de Audio
            # Usamos upload_file para subida directa
            print(f"Subiendo {filename} a {remote_path}...")
            self.api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type=self.dataset_type
            )
            
            # 3. Actualizar Metadata (metadata.csv)
            # El formato estándar de HF Audio Datasets es file_name,transcription
            self._append_to_remote_csv(remote_path, text)
            
            return True, f"✅ Subido a HF: {remote_path}"
            
        except Exception as e:
            return False, f"❌ Error subiendo a nube: {str(e)}"

    def _append_to_remote_csv(self, file_path, text):
        """
        Descarga el metadata.csv, añade una línea y lo vuelve a subir.
        Nota: Esto no es lo más eficiente para concurrencia masiva, pero sirve para prototipos.
        """
        csv_path = "metadata.csv"
        
        # Verificar si existe
        exists = self.fs.exists(f"datasets/{self.repo_id}/{csv_path}")
        
        lines = []
        if exists:
            # Leer contenido actual
            with self.fs.open(f"datasets/{self.repo_id}/{csv_path}", "r") as f:
                content = f.read()
                lines = content.strip().split("\n")
        else:
            # Crear cabecera
            lines.append("file_name,transcription")
            
        # Añadir nueva línea
        # file_path en metadata debe ser relativo a la raíz del dataset o absoluto? 
        # Para datasets de audio folder, suele ser solo el nombre si está en data/.
        # Dejaremos la ruta completa relativa "data/audio.wav" por seguridad.
        new_line = f"{file_path},{text}"
        lines.append(new_line)
        
        # Reconstruir CSV
        csv_content = "\n".join(lines)
        
        # Subir (Sobrescribir)
        # Usamos upload_file con un buffer en memoria
        csv_buffer = io.BytesIO(csv_content.encode("utf-8"))
        self.api.upload_file(
            path_or_fileobj=csv_buffer,
            path_in_repo=csv_path,
            repo_id=self.repo_id,
            repo_type=self.dataset_type
        )
        print("Metadata actualizado en la nube.")

    def download_audio(self, remote_filename, local_path, relative_folder="data"):
        """
        Descarga un archivo de audio desde Hugging Face.
        """
        try:
            from huggingface_hub import hf_hub_download
            
            remote_path = f"{relative_folder}/{remote_filename}"
            print(f"⬇️ Descargando {remote_path}...")
            
            # hf_hub_download descarga a cache, pero podemos moverlo o usar local_dir.
            # Usando local_dir y local_dir_use_symlinks=False para tener el archivo real.
            
            # Nota: hf_hub_download descarga toda la estructura de carpetas si usamos local_dir.
            # Ejemplo: local_dir/data/archivo.wav
            # Nosotros queremos el archivo final en 'local_path'.
            
            downloaded_path = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                repo_type=self.dataset_type,
                local_dir=".", # Descargar temporalmente en root para mantener estructura o manejarlo manualmente
                local_dir_use_symlinks=False
            )
            
            # El archivo descargado estará en ./data/filename.wav (o donde diga remote_path)
            # Necesitamos moverlo a local_path si es diferente.
            
            # Sin embargo, hf_hub_download returna la ruta absoluta del archivo.
            # Si usamos local_dir=".", creará la carpeta "data" en el root del proyecto.
            # Vamos a intentar leerlo y escribirlo en el destino final para evitar estructuras no deseadas,
            # o simplemente movemos el archivo.
            
            import shutil
            
            # Asegurar directorio destino
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Mover/Copiar al destino final
            shutil.move(downloaded_path, local_path)
            
            # Limpiar carpeta 'data' si quedó vacía (opcional, cuidado con borrar cosas útiles)
            # En este caso, como 'downloaded_path' apunta al archivo descargado, y lo movimos, 
            # ya está en su lugar.
            
            return True, f"✅ Descargado: {local_path}"
            
        except Exception as e:
            return False, f"❌ Error descargando: {str(e)}"

# Instancia global
try:
    cloud_handler = CloudHandler()
except Exception as e:
    print(f"Advertencia: No se pudo inicializar CloudHandler: {e}")
    cloud_handler = None
