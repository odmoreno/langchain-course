"""
ingest_modern.py
----------------
Ingesta 'moderna' de documentos a Pinecone usando LangChain v0.3.x,
Google AI Studio para embeddings, y buenas prácticas de batching.

Resumen del flujo:
1) Cargar -> 2) Split -> 3) Embeddings (Google) -> 4) Upsert en Pinecone

Por qué así:
- Google Generative AI Embeddings (`text-embedding-004`) devuelven 768 dims.
- Pinecone index debe crearse con dimension=768 y metric='cosine'.
- Usamos batching pequeño para evitar timeouts (504).

Cómo ejecutar:
$ python ingest_modern.py

Requisitos .env:
GOOGLE_API_KEY=...
PINECONE_API_KEY=...
INDEX_NAME=...
PINECONE_REGION=us-east-1
PINECONE_CLOUD=aws
"""

import os
import sys
import time
from typing import List

from dotenv import load_dotenv

# 1) Loader (lee el archivo de texto)
from langchain_community.document_loaders import TextLoader

# 2) Splitter moderno (mejor partición de texto)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 3) Embeddings con Google AI Studio (no pasar API key aquí; va por env)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 4) Vector store Pinecone (wrapper actual de LangChain)
from langchain_pinecone import PineconeVectorStore

# (Opcional) Crear índice si no existe
from pinecone import Pinecone, ServerlessSpec
from google.api_core.exceptions import DeadlineExceeded


# ---------------------------
# Parámetros "tuneables"
# ---------------------------
CHUNK_SIZE = 1000  # tamaño de chunk (caracteres)
CHUNK_OVERLAP = 150  # overlap entre chunks
EMBEDDING_MODEL = "text-embedding-004"  # 768 dims
EMBED_DIMS = 768
# Tamaños de lote pequeños y estables:
UPsert_BATCH_SIZE = 4  # upsert a Pinecone (por llamada)
# cuántos textos se envían al endpoint de embeddings por lote
EMBEDDING_CHUNK_SIZE = 4
# Reintentos simples ante 504 (picos de latencia de Google)
RETRIES = 4
BACKOFF_BASE = 2  # backoff exponencial (2^intento)


def ensure_index(index_name: str) -> None:
    """
    Crea el índice de Pinecone si no existe.
    Por claridad y portabilidad (dev/prod/CI), lo dejamos en el script.
    """
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    existing = [x["name"] for x in pc.list_indexes()]
    if index_name in existing:
        return

    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")

    print(
        f"[Pinecone] Creando índice '{index_name}' (dim={EMBED_DIMS}, metric=cosine, {cloud}/{region}) ..."
    )
    pc.create_index(
        name=index_name,
        dimension=EMBED_DIMS,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    # Los índices tardan unos segundos en estar disponibles.
    # Aquí un pequeño wait "amigable" (mejor que fallar por prisa).
    time.sleep(8)
    print("[Pinecone] Índice listo.")


def load_and_split(path: str):
    """
    Carga el archivo y lo divide en chunks 'saludables'.
    RecursiveCharacterTextSplitter prioriza cortar por saltos de línea/palabras.
    """
    loader = TextLoader(path, encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],  # orden de preferencia al cortar
    )
    chunks = splitter.split_documents(docs)
    return chunks


def build_embeddings():
    """
    Construye el objeto de embeddings de Google.
    Importante:
    - NO pasar google_api_key=..., la toma de GOOGLE_API_KEY en el entorno.
    - text-embedding-004 => 768 dims (alineado con el índice).
    - output_dimensionality=768 por claridad explícita.
    """
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        output_dimensionality=EMBED_DIMS,
    )


def upsert_with_retries(
    store: PineconeVectorStore, texts: List[str], metadatas: List[dict]
) -> None:
    """
    Realiza el upsert en lotes pequeños y con reintentos ante 504.
    Aprovecha que el wrapper moderno acepta:
      - batch_size            (upsert a Pinecone)
      - embedding_chunk_size  (lote hacia el endpoint de embeddings)
    Si tu versión no soporta esos kwargs, puedes:
      - Actualizar 'langchain-pinecone', o
      - Hacer un bucle manual que embeba y llame a add_vectors.
    """
    for attempt in range(RETRIES):
        try:
            store.from_texts(  # classmethod; se usa en modo "helper"
                texts=texts,
                embedding=build_embeddings(),  # nuevo objeto por seguridad de reintento
                metadatas=metadatas,
                index_name=os.environ["INDEX_NAME"],
                batch_size=UPsert_BATCH_SIZE,
                embedding_chunk_size=EMBEDDING_CHUNK_SIZE,
            )
            return
        except DeadlineExceeded as e:
            wait = BACKOFF_BASE**attempt
            print(
                f"[WARN] 504 DeadlineExceeded (intento {attempt+1}/{RETRIES}). Reintentando en {wait}s ..."
            )
            time.sleep(wait)
            last = e
        except TypeError as e:
            # Fallback si tu versión NO soporta esos kwargs (embedding_chunk_size/batch_size)
            print(
                "[INFO] Tu versión de langchain-pinecone no soporta 'embedding_chunk_size'/'batch_size' en from_texts(). "
                "Aplicando fallback: bucle manual (embeddings + add_vectors)."
            )
            manual_upsert(texts, metadatas)
            return
    # Si llegamos aquí, agotamos reintentos
    raise last


def manual_upsert(texts: List[str], metadatas: List[dict]) -> None:
    """
    Fallback universal (funciona con cualquier versión):
    - Calcula embeddings en micro-lotes.
    - Inserta con add_vectors (evita que el wrapper los calcule por dentro).
    """
    embeddings = build_embeddings()
    store = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )

    ids: List[str] = []
    vecs: List[List[float]] = []
    txs: List[str] = []
    metas: List[dict] = []

    # Micro-batching estable para evitar 504:
    for i in range(0, len(texts), EMBEDDING_CHUNK_SIZE):
        sub_texts = texts[i : i + EMBEDDING_CHUNK_SIZE]
        sub_metas = metadatas[i : i + EMBEDDING_CHUNK_SIZE]

        # Reintentos mínimos ante 504 de Google:
        for attempt in range(RETRIES):
            try:
                sub_vecs = embeddings.embed_documents(sub_texts)
                break
            except DeadlineExceeded:
                wait = BACKOFF_BASE**attempt
                print(
                    f"[WARN] 504 al embeddear (intento {attempt+1}/{RETRIES}). Esperando {wait}s ..."
                )
                time.sleep(wait)
        else:
            raise RuntimeError("Fallo repetido al embeddear con Google (504).")

        # Acumular para un único add_vectors grande o varios medianos
        start_id = len(ids)
        ids.extend([f"doc-{start_id + j}" for j in range(len(sub_texts))])
        vecs.extend(sub_vecs)
        txs.extend(sub_texts)
        metas.extend(sub_metas)

        # Upsert en lotes chicos hacia Pinecone (opcional):
        if len(vecs) >= 32:  # flush oportunista
            store.add_vectors(vectors=vecs, ids=ids, texts=txs, metadatas=metas)
            ids, vecs, txs, metas = [], [], [], []

    # Flush final:
    if vecs:
        store.add_vectors(vectors=vecs, ids=ids, texts=txs, metadatas=metas)


def main():
    load_dotenv()

    index_name = os.environ.get("INDEX_NAME")
    if not index_name:
        print("[ERROR] Falta INDEX_NAME en .env")
        sys.exit(1)

    # 0) (Opcional) crear el índice si no existe
    ensure_index(index_name)

    # 1) Cargar y dividir
    print("Cargando y partiendo documento ...")
    chunks = load_and_split("mediumblog1.txt")
    print(f"Chunks creados: {len(chunks)}")

    texts = [d.page_content for d in chunks]
    metadatas = [d.metadata for d in chunks]

    # 2) Vector store "handle" (se puede crear vacío; .from_texts es classmethod helper)
    embeddings = build_embeddings()
    store = PineconeVectorStore(index_name=index_name, embedding=embeddings)

    # 3) Upsert con control de lote y reintentos
    print("Inyectando en Pinecone (con batching y reintentos) ...")
    upsert_with_retries(store, texts, metadatas)

    print("✔ Ingesta finalizada sin errores.")


if __name__ == "__main__":
    main()
