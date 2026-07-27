import os
import json
import time
from dotenv import load_dotenv
import replicate

load_dotenv()

with open("marcos_references.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

ref_info = catalog["references"][0]
print(f"Foto de referência real selecionada: {ref_info['file_name']}")
orig_img_path = ref_info["path"]

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

file_obj = client.files.create(open(orig_img_path, "rb"))
real_photo_url = file_obj.urls["get"]

# Prompt negativo e diretivas explícitas contra sardas, manchas e alisamento digital
prompt = (
    "A full-body high-resolution professional studio portrait of the man from the reference image (Marcos). "
    "He is sitting relaxed on an elegant leather armchair in a modern executive office, smiling naturally. "
    "EXACT INSTRUCTION FOR SKIN: Maintain 100% natural authentic male skin texture with realistic visible micro-pores, natural skin grain, and smooth uniform skin tone. "
    "DO NOT ADD FRECKLES. DO NOT ADD SPOTS. NO FRECKLES, NO MOLES, NO BLEMISHES. "
    "DO NOT AIRBRUSH OR SMOOTH THE SKIN. NO PLASTIC WAX EFFECT. "
    "Keep 100% exact facial identity, real beard density, and chest hair from the reference photo. "
    "Hyper-realistic 8k photography, 35mm lens, natural studio lighting."
)

print("\nDisparando geração Nano Banana Pro com calibração de pele (Sem sardas, sem alisamento)...")
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
    print("\nIMAGEM COM PELE CALIBRADA GERADA COM SUCESSO:")
    print("URL:", pred.output)
else:
    print("\nERRO:", pred.error)
