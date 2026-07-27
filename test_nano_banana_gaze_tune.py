import os
import json
import time
from dotenv import load_dotenv
import replicate

load_dotenv()

# Carregar o preset base aprovado
with open("preset_marcos_approved.json", "r", encoding="utf-8") as f:
    preset = json.load(f)

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

file_obj = client.files.create(open("G:/Meu Drive/Imagens/Exemplos de Marcos/Marcos 01 leve sorriso.png", "rb"))
real_photo_url = file_obj.urls["get"]

# Adicionamos ao prompt perfeito as diretivas exatas de micro-calibração do olhar
prompt_gaze_tuned = (
    preset["exact_prompt"] +
    " EYE AND GAZE FINE-TUNING: Maintain the exact authentic expressive gaze of Marcos. "
    "Include the natural subtle brow asymmetry (left eyebrow slightly higher and more expressive), "
    "deep dark iris with sharp catchlight reflections, and the slight natural muscle contraction of the lower eyelids when smiling. "
    "Warm, engaging, energetic, and highly expressive human eyes."
)

print("\nDisparando geração Nano Banana Pro com calibração de micro-olhar (Preset Aprovado + Gaze Tuning)...")
pred = client.predictions.create(
    version=preset["model_version"],
    input={
        "prompt": prompt_gaze_tuned,
        "image_input": [real_photo_url],
        "aspect_ratio": "1:1",
        "output_format": "png"
    }
)

print(f"Predição iniciada! ID: {pred.id}")
start = time.time()
while pred.status not in ["succeeded", "failed"]:
    time.sleep(3)
    pred.reload()
    print(f"Status ({int(time.time() - start)}s): {pred.status}")

if pred.status == "succeeded":
    print("\nIMAGEM COM CALIBRACAO DE OLHAR GERADA COM SUCESSO:")
    print("URL:", pred.output)
else:
    print("\nERRO:", pred.error)
