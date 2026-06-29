"""Sistema RAG para cargar documentos Markdown, vectorizarlos y recuperar contexto relevante."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RAGSystem:
    """Gestiona embeddings, almacenamiento vectorial en memoria y búsqueda semántica."""

    def __init__(
        self,
        documents_path: str | Path = "documents",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 900,
        chunk_overlap: int = 150,
        default_k: int = 4,
    ) -> None:
        self.documents_path = Path(documents_path)
        self.embedding_model = embedding_model
        self.default_k = default_k
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        )
        self.raw_documents: List[Document] = []
        self.document_chunks: List[Document] = []
        self.index_ready = False

    def load_markdown_documents(self) -> List[Document]:
        """Carga archivos Markdown desde la carpeta configurada y conserva metadatos de origen."""
        if not self.documents_path.exists():
            raise FileNotFoundError(f"No existe la carpeta de documentos: {self.documents_path}")

        markdown_files = sorted(self.documents_path.glob("*.md"))
        if not markdown_files:
            raise FileNotFoundError(f"No se encontraron archivos Markdown en: {self.documents_path}")

        documents: List[Document] = []
        for file_path in markdown_files:
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": file_path.name,
                        "path": str(file_path),
                    },
                )
            )

        if not documents:
            raise ValueError("Los documentos Markdown encontrados no contienen texto procesable.")

        self.raw_documents = documents
        return documents

    def split_documents(self, documents: Optional[Iterable[Document]] = None) -> List[Document]:
        """Divide documentos en fragmentos adecuados para recuperación semántica."""
        source_documents = list(documents) if documents is not None else self.raw_documents
        if not source_documents:
            source_documents = self.load_markdown_documents()

        chunks = self.text_splitter.split_documents(source_documents)
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = index

        self.document_chunks = chunks
        return chunks

    def build_index(self) -> None:
        """Crea el índice vectorial en memoria con embeddings de OpenAI."""
        documents = self.load_markdown_documents()
        chunks = self.split_documents(documents)
        self.vector_store.add_documents(chunks)
        self.index_ready = True

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Recupera los fragmentos más relevantes para una consulta."""
        if not self.index_ready:
            self.build_index()

        top_k = k or self.default_k
        return self.vector_store.similarity_search(query, k=top_k)

    @staticmethod
    def format_documents(documents: Iterable[Document]) -> str:
        """Construye un bloque de contexto con contenido y metadatos de cada fragmento."""
        formatted_chunks = []
        for position, document in enumerate(documents, start=1):
            source = document.metadata.get("source", "fuente_desconocida")
            chunk_id = document.metadata.get("chunk_id", "sin_id")
            formatted_chunks.append(
                f"[Fragmento {position} | fuente={source} | chunk={chunk_id}]\n"
                f"{document.page_content}"
            )
        return "\n\n---\n\n".join(formatted_chunks)
