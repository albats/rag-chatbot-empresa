"""Lógica conversacional que combina recuperación documental y generación con OpenAI."""

from __future__ import annotations

from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .rag_system import RAGSystem


class Chatbot:
    """Chatbot conversacional con historial de sesión y respuestas basadas en documentos."""

    def __init__(
        self,
        rag_system: RAGSystem,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        max_history_messages: int = 10,
    ) -> None:
        self.rag_system = rag_system
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.max_history_messages = max_history_messages
        self.history: List[Dict[str, str]] = []

    def ask(self, query: str) -> str:
        """Ejecuta el flujo RAG completo: consulta, retrieval, generación y respuesta."""
        retrieved_documents = self.rag_system.retrieve(query)
        context = self.rag_system.format_documents(retrieved_documents)
        history_text = self._format_history()

        messages = [
            SystemMessage(
                content=(
                    "Eres un asistente interno de NovaMercurio Solutions. "
                    "Responde en español y utiliza únicamente la información presente en el contexto recuperado. "
                    "Puedes usar el historial solo para entender referencias conversacionales, no para inventar datos. "
                    "Si el contexto no contiene información suficiente para responder, indica que no hay información suficiente en los documentos procesados. "
                    "Incluye al final una línea breve con las fuentes Markdown usadas."
                )
            ),
            HumanMessage(
                content=(
                    f"Historial de conversación de la sesión:\n{history_text}\n\n"
                    f"Contexto recuperado de documentos Markdown:\n{context}\n\n"
                    f"Pregunta actual:\n{query}"
                )
            ),
        ]

        response = self.llm.invoke(messages)
        answer = str(response.content).strip()

        self._append_history("user", query)
        self._append_history("assistant", answer)
        return answer

    def _append_history(self, role: str, content: str) -> None:
        """Mantiene una ventana limitada del historial conversacional."""
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history_messages:
            self.history = self.history[-self.max_history_messages :]

    def _format_history(self) -> str:
        """Convierte el historial de la sesión en texto para el prompt."""
        if not self.history:
            return "Sin historial previo."

        return "\n".join(
            f"{message['role']}: {message['content']}" for message in self.history
        )

    def reset_history(self) -> None:
        """Limpia el historial conversacional de la sesión actual."""
        self.history.clear()
