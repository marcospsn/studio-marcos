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

# Prompt calibrado: 100% sem sardas + amenização suave das linhas de expressão (mantendo o aspecto jovem de 35 anos)
prompt = (
    "A full-body high-resolution professional studio portrait of the man from the reference image (Marcos, a handsome 35yo man). "
    "He is sitting relaxed on an elegant leather armchair in a modern executive office, smiling naturally. "
    "SKIN AND FACE SPECIFICATIONS: Perfectly spotless, smooth, clean olive skin complexion without any freckles. "
    "ZERO FRECKLES. COMPLETELY FRECKLE-FREE FACE. ZERO SUN SPOTS. ZERO SPOTS. "
    "AGE AND WRINKLE ADJUSTMENT: Youthful smooth skin for a 35yo man with very soft, subtle, minimal expression lines around the eyes. "
    "REDUCE EYE WRINKLES AND FOREHEAD LINES BY 60%. Soften deep facial creases to look youthful, fresh, and energetic while keeping natural micro-pores. "
    "Keep 100% exact facial identity, excellent dense beard, realistic teeth, styled hair, and natural chest hair from the reference photo. "
    "Hyper-realistic 8k photography, 35mm lens, natural studio lighting."
)

print("\nDisparando geração Nano Banana Pro (Pele 100% sem sardas + Rugas amenizadas para 35 anos)...")
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
    print("\nIMAGEM PERFEITA GERADA COM SUCESSO:")
    print("URL:", pred.output)
else:
    print("\nERRO:", pred.error)
