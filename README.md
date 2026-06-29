# Chatbot RAG con LangChain y OpenAI

## Descripción

Proyecto de Python que implementa un chatbot inteligente con técnica RAG, Retrieval-Augmented Generation, para responder preguntas basadas en documentos Markdown de una empresa ficticia llamada NovaMercurio Solutions. El sistema carga documentos internos, los divide en fragmentos, genera embeddings con OpenAI, almacena los vectores en memoria y recupera el contexto más relevante para generar respuestas conversacionales.

## Estructura del proyecto

```text
├── main.py
├── documents/
│   ├── documento1.md
│   └── documento2.md
├── core/
│   ├── __init__.py
│   ├── rag_system.py
│   └── chatbot.py
├── requirements.txt
├── .env
└── README.md
```

## Dependencias

Las dependencias principales son LangChain, LangChain OpenAI, LangChain Community, LangChain Text Splitters, OpenAI y python-dotenv. El archivo `requirements.txt` contiene las librerías necesarias para ejecutar el proyecto.

## Configuración del entorno

Crear un entorno virtual e instalar dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configurar la clave de OpenAI en el archivo `.env`:

```env
OPENAI_API_KEY=tu_api_key_de_openai
```

## Ejecución

Iniciar el chatbot desde la raíz del proyecto:

```bash
python main.py
```

El programa inicializa el sistema RAG, procesa los documentos Markdown y abre una interfaz de línea de comandos. La sesión se termina con `/salir` o `quit`.

## Funcionamiento técnico

El archivo `core/rag_system.py` contiene la clase `RAGSystem`, responsable de cargar documentos Markdown, dividirlos en fragmentos con `RecursiveCharacterTextSplitter`, generar embeddings mediante `OpenAIEmbeddings` con el modelo `text-embedding-3-small` y almacenar los vectores en `InMemoryVectorStore`.

El archivo `core/chatbot.py` contiene la clase `Chatbot`, que integra el sistema de retrieval con un modelo conversacional de OpenAI. El modelo configurado es `gpt-4o`, con temperatura cero para favorecer respuestas consistentes. El flujo implementado es consulta, recuperación de fragmentos relevantes, construcción de contexto, generación de respuesta y actualización del historial de conversación.

El archivo `main.py` contiene la interfaz CLI, validación de la variable `OPENAI_API_KEY`, inicialización del índice vectorial, bucle de conversación, comandos de salida y manejo básico de errores de autenticación, conexión, límite de uso y errores inesperados.

## Documentos procesados

`documents/documento1.md` describe información general ficticia de NovaMercurio Solutions, incluyendo historia, misión, visión, valores, servicios, sectores atendidos, principios de inteligencia artificial, organización interna y diferenciadores.

`documents/documento2.md` describe políticas internas y procedimientos ficticios, incluyendo horarios, beneficios, código de conducta, seguridad de la información, gestión de proyectos, soporte, documentación, conflictos de interés y confidencialidad.

## Restricción de respuesta documental

El prompt del chatbot indica que las respuestas deben basarse únicamente en el contexto recuperado de los documentos procesados. Si los fragmentos recuperados no contienen información suficiente, el chatbot debe indicarlo en lugar de inventar información.
