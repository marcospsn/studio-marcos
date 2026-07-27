import os
import time
import replicate
from dotenv import load_dotenv

load_dotenv()

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
ZIP_PATH = r"D:\OneDrive\Documentos\PROJETOS\ARQUITETO\studio_marcos\marcos_dataset_captioned.zip"

if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN não foi encontrado no .env")

client = replicate.Client(api_token=REPLICATE_API_TOKEN)

print("1. Fazendo upload do novo dataset ZIP com legendas .txt...")
file_obj = client.files.create(open(ZIP_PATH, "rb"))
zip_url = file_obj.urls["get"]
print(f"Dataset hospedado com sucesso! URL: {zip_url}")

print("2. Disparando o treinamento oficial do LoRA do Marcos no Replicate...")
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

while training.status in ["starting", "processing"]:
    time.sleep(15)
    training.reload()
    print(f"Status do Treino: {training.status}...")

if training.status == "succeeded":
    print("\nTREINAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"Pesos gerados: {training.output}")
    with open("lora_config.json", "w", encoding="utf-8") as f:
        f.write(str(training.output))
else:
    print(f"\nErro no treinamento: {training.error}")
