# adapters.py
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings


class GoogleEmbeddingsSafe(Embeddings):
    """
    Adapter para asegurar que embed_query/embed_documents devuelvan listas nativas de Python.
    Evita errores de serialización ('Repeated') en el cliente de Pinecone.
    """

    def __init__(self, inner: GoogleGenerativeAIEmbeddings) -> None:
        self.inner = inner

    def embed_query(self, text: str) -> List[float]:
        vec = self.inner.embed_query(text)
        # si 'vec' es un 'Repeated' de protobuf, list(...) lo convierte a list[float]
        return list(vec)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        vecs = self.inner.embed_documents(texts)
        # asegúrate de que cada vector sea list[float]
        return [list(v) for v in vecs]
