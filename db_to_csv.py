import chromadb
import csv
import os
from pathlib import Path

def main():
    # --- AGREGAMOS ESTO PARA QUE USE LA MISMA RUTA QUE EL SISTEMA PRINCIPAL ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DB_PATH = os.path.join(SCRIPT_DIR, "af5_chroma_db")
    # ------------------------------------------------------------------------

    client = chromadb.PersistentClient(path=DB_PATH)
    
    # Usamos get_or_create para que no crashee si no existe (y la crea si hace falta)
    collection = client.get_or_create_collection("af5_nodes")
    
    print(f"Usando colección: {collection.name} en {DB_PATH}")
    result = collection.get(include=["metadatas"])
    ids = result["ids"]
    metadatas = result["metadatas"]
    
    if not ids:
        print("No hay ítems para exportar. Asegurate de haber ejecutado el chat antes.")
        return
    
    print(f"Exportando {len(ids)} ítems...")
    csv_path = os.path.join(SCRIPT_DIR, "db_export.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "name", "type", "agent_id", "creator_id",
            "social", "academic", "emotional", "aesthetic", "linguistic",
            "config_json", "weights", "context_memory", "related_entities"
        ])
                
        for idx, meta in zip(ids, metadatas):
            row = [
                idx,
                meta.get("name", idx),
                meta.get("type", ""),
                meta.get("agent_id", ""),
                meta.get("creator_id", ""),
                meta.get("social", ""),
                meta.get("academic", ""),
                meta.get("emotional", ""),
                meta.get("aesthetic", ""),
                meta.get("linguistic", ""),
                meta.get("config_json", ""),
                meta.get("weights", ""),
                meta.get("context_memory", ""),
                meta.get("related_entities", "")
            ]
            writer.writerow(row)
    
    print(f"✅ Exportación completada: {csv_path}")

if __name__ == "__main__":
    main()