import os
import time
import replicate
from dotenv import load_dotenv

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
ZIP_PATH = r"D:\OneDrive\Documentos\PROJETOS\ARQUITETO\studio_marcos\marcos_dataset.zip"

if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN não foi encontrado no .env")

client = replicate.Client(api_token=REPLICATE_API_TOKEN)

print("1. Fazendo upload do dataset ZIP para o servidor do Replicate...")
file_obj = client.files.create(open(ZIP_PATH, "rb"))
zip_url = file_obj.urls["get"]
print(f"Dataset hospedado com sucesso! URL: {zip_url}")

print("2. Disparando o treinamento do LoRA FLUX-dev no Replicate...")
training = client.trainings.create(
    version="replicate/fast-flux-trainer:e5a5bc821112c107e6ddb8491c5b898f94d06eaca861d1dbf37b29cd69ba8988",
    input={
        "input_images": zip_url,
        "trigger_word": "marcos_paulo",
        "steps": 1000,
        "lora_type": "subject"
    },
    destination="marcospsn/marcos-paulo-lora"
)

print(f"Treinamento iniciado! ID: {training.id}")
print(f"Status inicial: {training.status}")
print("Você pode acompanhar os logs no Replicate ou aguardar aqui no script.")

while training.status in ["starting", "processing"]:
    time.sleep(15)
    training.reload()
    print(f"Status atual: {training.status}...")

if training.status == "succeeded":
    print("\nPARABÉNS! Treinamento concluído com sucesso!")
    print(f"Modelo LoRA gerado: {training.output}")
    with open("lora_config.json", "w") as f:
        f.write(f'{{"lora_model": "{training.output.get("weights")}", "version": "{training.id}"}}')
else:
    print(f"\nErro no treinamento: {training.error}")

