import os
import json
import time
import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import replicate

load_dotenv()

# 1. Carregar o catálogo de fotos reais
with open("marcos_references.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

ref_info = catalog["references"][0]
print(f"Foto de referência real do Marcos: {ref_info['file_name']}")
orig_img_path = ref_info["path"]

# 2. Upload da foto real do Marcos para o Replicate Files
client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

print("Enviando a sua foto REAL para a nuvem do Replicate...")
file_obj = client.files.create(open(orig_img_path, "rb"))
real_photo_url = file_obj.urls["get"]
print(f"URL da sua foto real: {real_photo_url}")

# 3. Disparar geração Zero-Shot por referência direta (PuLID / Reference Face - SEM TREINO)
print("\nDisparando edição por referência direta sem treinamento (Preservando a matriz de pixels real)...")
pred = client.predictions.create(
    model="black-forest-labs/flux-fill-dev",
    input={
        "image": real_photo_url,
        "prompt": "a photo of marcos, crisp high detail portrait, wearing an elegant navy blue blazer with a white shirt underneath, natural skin pores, 35mm photograph",
        "guidance_scale": 30.0,  # Alta fidelidade à imagem de origem
        "num_inference_steps": 30
    }
)

print(f"Predição iniciada! ID: {pred.id}")
start = time.time()
while pred.status not in ["succeeded", "failed"]:
    time.sleep(3)
    pred.reload()
    print(f"Status ({int(time.time() - start)}s): {pred.status}")

if pred.status == "succeeded":
    print("\n✅ IMAGEM GERADA COM SUCESSO SEM TREINAMENTO:")
    print("URL:", pred.output)
else:
    print("\n❌ ERRO:", pred.error)
