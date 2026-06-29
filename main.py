"""Punto de entrada de la aplicación de línea de comandos para el chatbot RAG."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, AuthenticationError, OpenAIError, RateLimitError

from core import Chatbot, RAGSystem


def create_chatbot() -> Chatbot:
    """Inicializa el sistema RAG, vectoriza documentos Markdown y prepara el chatbot."""
    project_root = Path(__file__).resolve().parent
    documents_path = project_root / "documents"

    rag_system = RAGSystem(documents_path=documents_path)
    rag_system.build_index()
    return Chatbot(rag_system=rag_system, model_name="gpt-4o")


def run_cli() -> None:
    """Ejecuta el bucle principal de conversación por consola."""
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        print("Falta la variable OPENAI_API_KEY en el archivo .env o en el entorno.")
        return

    print("Inicializando sistema RAG con documentos Markdown...")

    try:
        chatbot = create_chatbot()
    except Exception as error:
        print(f"No se pudo inicializar el sistema RAG: {error}")
        return

    print("Chatbot listo. Escribe una pregunta sobre NovaMercurio Solutions.")
    print("Comandos disponibles: /salir, quit")

    while True:
        try:
            query = input("\nConsulta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSesión finalizada.")
            break

        if not query:
            continue

        if query.lower() in {"/salir", "quit"}:
            print("Sesión finalizada.")
            break

        try:
            answer = chatbot.ask(query)
            print(f"\nRespuesta:\n{answer}")
        except AuthenticationError:
            print("Error de autenticación con OpenAI. Revisa OPENAI_API_KEY.")
        except RateLimitError:
            print("Límite de uso alcanzado en la API de OpenAI. Intenta más tarde.")
        except APIConnectionError:
            print("Error de conexión con OpenAI. Revisa la conexión de red.")
        except OpenAIError as error:
            print(f"Error del servicio de OpenAI: {error}")
        except Exception as error:
            print(f"Error inesperado durante la consulta: {error}")


if __name__ == "__main__":
    run_cli()
