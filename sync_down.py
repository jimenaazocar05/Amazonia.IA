import os
import csv
from cloud_handler import cloud_handler

def sync_down():
    print("="*40)
    print("🔄 INICIANDO SINCRONIZACIÓN (CLOUD -> LOCAL)")
    print("="*40)

    if not cloud_handler:
        print("❌ Error: No se pudo inicializar el CloudHandler (revisar .env)")
        return

    # Rutas
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "datos_finales", "nuevos_datos.csv")
    
    # 1. Leer CSV (Source of Truth)
    if not os.path.exists(csv_path):
        print(f"❌ Error: No existe el CSV maestro en {csv_path}")
        return

    print(f"📂 CSV Maestro: {csv_path}")
    
    processed = 0
    restored = 0
    errors = 0
    skipped = 0

    try:
        with open(csv_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        total = len(rows)
        print(f"📊 Registros totales en CSV: {total}")
        
        for row in rows:
            processed += 1
            csv_path_rel = row.get("audio_path", "")
            
            if not csv_path_rel:
                continue
                
            # Construir ruta absoluta local esperada
            # El CSV suele tener 'nuevos_audios_wav\archivo.wav' o 'nuevos_audios_wav/archivo.wav'
            # Normalizamos separadores para el OS actual
            local_full_path = os.path.join(base_dir, "datos_finales", csv_path_rel.replace("\\", "/"))
            filename = os.path.basename(local_full_path)
            
            # Verificar existencia local
            if os.path.exists(local_full_path):
                # print(f"✅ OK (Local): {filename}")
                continue
            
            print(f"⚠️ FALTANTE detectado: {filename}")
            
            # Intentar descargar
            # Asumimos que en HF están en 'data/'
            success, msg = cloud_handler.download_audio(
                remote_filename=filename,
                local_path=local_full_path,
                relative_folder="data"
            )
            
            if success:
                print(f"   ╚══ {msg}")
                restored += 1
            else:
                print(f"   ╚══ {msg}")
                # Si falla es probable que no exista en la nube tampoco (error original de subida)
                if "404" in msg or "Entry not found" in msg:
                     print("      (Este archivo no existe en la nube, posible error de subida original)")
                errors += 1

    except Exception as e:
        print(f"❌ Error crítico en script: {e}")

    print("\n" + "="*40)
    print("🏁 RESUMEN DE SINCRONIZACIÓN")
    print("="*40)
    print(f"Total procesados: {processed}")
    print(f"✅ Archivos OK (Ya existían): {processed - restored - errors}")
    print(f"📥 Restaurados de Nube:       {restored}")
    print(f"❌ Errores (No en nube/Otros): {errors}")

    if errors > 0:
        print("\nNOTA: Los errores suelen ser archivos que se guardaron en el CSV pero falló su subida inicial.")

if __name__ == "__main__":
    sync_down()
