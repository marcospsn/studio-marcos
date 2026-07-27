import os
import time
import replicate
from dotenv import load_dotenv

load_dotenv()

client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))

# Novo LoRA treinado pelo Ostris AI-Toolkit oficial em alta fidelidade
lora_weights = "https://replicate.delivery/xezq/igNCg7q3rw6lEFHGsLKnM8PkauonTwx0N6OaoRPEGzVE6EvF/trained_model.tar"

print("Disparando predição de alta fidelidade...")
pred = client.predictions.create(
    model="black-forest-labs/flux-dev",
    input={
        "prompt": "a photo of marcos_paulo, studio portrait, dark hair, short beard, natural skin texture with detailed pores, 35mm photography",
        "extra_lora": lora_weights,
        "extra_lora_scale": 0.85,
        "guidance_scale": 3.5,
        "num_inference_steps": 28
    }
)

print("ID da Predição:", pred.id)
start = time.time()
while pred.status not in ["succeeded", "failed"]:
    time.sleep(3)
    pred.reload()
    print(f"Status ({int(time.time() - start)}s): {pred.status}")

if pred.status == "succeeded":
    print("\nIMAGEM GERADA COM SUCESSO:")
    print("URL:", pred.output)
else:
    print("\nERRO NA GERACAO:", pred.error)
