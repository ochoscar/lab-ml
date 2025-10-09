import pytesseract
from pdf2image import convert_from_path
import subprocess
import json

# Convierte PDF a imágenes
pages = convert_from_path(r"C:\Users\ochos\OneDrive\Documentos\factura.pdf", dpi=300)


# Extrae texto de la primera página (o iterar todas)
texto = pytesseract.image_to_string(pages[0], lang="spa")

#print(texto)

def preguntar_llm(prompt, model="llama3"):
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8")

# Prompt para extracción estructurada
prompt = f"""
Extrae los siguientes campos del texto:
- Propietario
- Dir. Entrega

Texto:
{texto}

Responde ÚNICAMENTE con un JSON válido en este formato:
{{
  "propietario": "...",
  "dir_entrega": "..."
}}
No incluyas explicaciones, ni comentarios, ni texto adicional.
"""

print("Prompt enviado al LLM")
respuesta = preguntar_llm(prompt)
print(respuesta)

try:
    datos = json.loads(respuesta)
    print("Propietario:", datos.get("propietario"))
    print("Dir. Entrega:", datos.get("dir_entrega"))
except:
    print("La respuesta no es JSON válido:", respuesta)
