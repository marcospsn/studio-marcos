import os
import json
import time
from dotenv import load_dotenv
import replicate

load_dotenv()

# Carregar o catálogo de fotos reais do Marcos
with open("marcos_references.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

ref_info = catalog["references"][0]
print(f"Foto de referência real selecionada: {ref_info['file_name']}")
orig_img_path = ref_info["path"]

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

print("Fazendo o upload da foto real do Marcos...")
file_obj = client.files.create(open(orig_img_path, "rb"))
real_photo_url = file_obj.urls["get"]

prompt = (
    "A full-body high-resolution professional studio portrait of the man from the reference image (Marcos). "
    "He is sitting relaxed on an elegant leather armchair in a modern executive office, smiling naturally. "
    "Keep 100% exact facial identity, real skin pores, hair style, chest hair, and beard density from the reference photo. "
    "Hyper-realistic 8k photography, 35mm lens, natural studio lighting."
)

print("\nDisparando geração do zero com NANO BANANA PRO (Gemini 3 Pro Image) via Replicate...")
pred = client.predictions.create(
    version="google/nano-banana-pro:93f55bfdbdfd4a62e16bf861729bcfa9e8fd9b0325fb218cbc4dd138ecc87cc7",
    input={
        "prompt": prompt,
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
    print("\n✅ IMAGEM GERADA COM SUCESSO PELO NANO BANANA PRO:")
    print("URL:", pred.output)
else:
    print("\n❌ ERRO NO NANO BANANA PRO:", pred.error)
