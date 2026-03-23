"""
Pipeline completo para analizar un PDF con la API de OpenAI.

Pasos:
1. Sube el PDF a OpenAI
2. Crea un vector store e indexa el documento (lo hace buscable semanticamente)
3. Lanza la consulta con la Responses API y obtiene la respuesta
"""
from dotenv import load_dotenv # python-dotenv
import os
import json
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("No se encontro la API key en el .env")

client = OpenAI(api_key=api_key)

print("Cliente inicializado correctamente")

# Paso 1: subir el PDF. OpenAI almacena el archivo y devuelve un ID unico para referenciarlo.
file = client.files.create(
    file=open("documento.pdf", "rb"),
    purpose="assistants"
)

# Paso 2: crear el vector store e indexar el archivo.
# El PDF se divide en fragmentos, se convierte en embeddings y queda buscable semanticamente.
vector_store = client.vector_stores.create(name="Documentos del proyecto")
client.vector_stores.files.create_and_poll(
    vector_store_id=vector_store.id,
    file_id=file.id
)

# Paso 3: lanzar la consulta. La Responses API busca en el vector store y devuelve un JSON estructurado.
# `json_object` obliga al modelo a devolver siempre JSON valido, con la estructura que mejor refleje el documento.
response = client.responses.create(
    model="gpt-4o",
    input="Extrae toda la informacion estructurada de este documento: tablas, datos puntuales, campos clave y sus valores. Devuelve un JSON que refleje fielmente el contenido.",
    instructions="Eres un experto en extraccion de datos de documentos. Devuelve UNICAMENTE un JSON valido, sin texto adicional. Organiza el contenido de forma logica: tablas como listas de objetos, campos simples como pares clave-valor.",
    text={"format": {"type": "json_object"}},
    tools=[{
        "type": "file_search",
        "vector_store_ids": [vector_store.id]
    }]
)

resultado = json.loads(response.output_text)
print("\n--- Respuesta del modelo ---")
print(json.dumps(resultado, indent=2, ensure_ascii=False))