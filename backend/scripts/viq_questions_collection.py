#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import chromadb


COLLECTION_NAME = "viq_questions"
_REPO_ROOT = Path(__file__).parent.parent.parent  # data/ is a sibling of backend/
DEFAULT_DB_PATH = str(_REPO_ROOT / "data" / "vectordb")
DEFAULT_EXPORT_PATH = str(_REPO_ROOT / "data" / "viq_questions_export.json")
BATCH_SIZE = 100


def _as_jsonable_embedding(embedding):
    if embedding is None:
        return None
    if hasattr(embedding, "tolist"):
        return embedding.tolist()
    return list(embedding)


def export_viq_questions(db_path: str, output_path: str) -> None:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas", "embeddings"])

    records = []
    for item_id, document, metadata, embedding in zip(
        data.get("ids", []),
        data.get("documents", []),
        data.get("metadatas", []),
        data.get("embeddings", []),
    ):
        records.append(
            {
                "id": item_id,
                "document": document,
                "metadata": metadata or {},
                "embedding": _as_jsonable_embedding(embedding),
            }
        )

    export_data = {
        "collection_name": COLLECTION_NAME,
        "collection_metadata": collection.metadata or {},
        "count": len(records),
        "records": records,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export_data), encoding="utf-8")
    print(f"Exported {len(records)} records to {output}")


def import_viq_questions(db_path: str, input_path: str) -> None:
    payload = json.loads(Path(input_path).read_text(encoding="utf-8"))
    records = payload.get("records", [])

    client = chromadb.PersistentClient(path=db_path)
    existing_names = [collection.name for collection in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata=payload.get("collection_metadata") or {"hnsw:space": "cosine"},
    )

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        collection.add(
            ids=[record["id"] for record in batch],
            documents=[record["document"] for record in batch],
            metadatas=[record["metadata"] for record in batch],
            embeddings=[record["embedding"] for record in batch],
        )

    print(f"Imported {len(records)} records into {COLLECTION_NAME}")
    print("Other collections, including viq_training, were not modified.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export or import only the viq_questions Chroma collection."
    )
    parser.add_argument("action", choices=["export", "import"])
    parser.add_argument("--db-path", default=DEFAULT_DB_PATH)
    parser.add_argument("--file", default=DEFAULT_EXPORT_PATH)
    args = parser.parse_args()

    if args.action == "export":
        export_viq_questions(args.db_path, args.file)
    else:
        import_viq_questions(args.db_path, args.file)


if __name__ == "__main__":
    main()
