# Treinamento de LoRA FLUX.1 (Alta Fidelidade) no Google Colab
# Este notebook automatiza a instalação do AI-Toolkit, configuração do dataset do Marcos e treinamento profissional de 1024x1024.

import os

print("=== STUDIO MARCOS - SETUP DO TREINAMENTO DE LORA ===")

# 1. Montar Google Drive para salvar o resultado final diretamente no seu Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clonar o AI-Toolkit (Framework oficial SOTA para FLUX LoRA)
!git clone https://github.com/ostris/ai-toolkit.git
%cd ai-toolkit
!git submodule update --init --recursive
!pip install -r requirements.txt
!pip install lycoris_lora diffusers transformers ftfy accelerate

# 3. Copiar e extrair o dataset legendado do Google Drive
DATASET_ZIP = "/content/drive/MyDrive/Imagens/Exemplos de Marcos/marcos_dataset_captioned.zip"
!mkdir -p /content/dataset_marcos
!unzip -o "$DATASET_ZIP" -d /content/dataset_marcos

# 4. Criar arquivo de configuração YAML de Alta Fidelidade (1024x1024 com preservação de poros)
config_yaml = """
job: extension
config:
  name: "marcos_paulo_lora_hd"
  process:
    - type: 'sd_trainer'
      training_folder: "/content/drive/MyDrive/LoRA_Marcos_Output"
      device: cuda:0
      trigger_word: "marcos_paulo"
      network:
        type: "lora"
        linear: 16
        linear_alpha: 16
      save:
        dtype: float16
        save_every: 250
        max_step_saves: 4
      datasets:
        - folder_path: "/content/dataset_marcos"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [1024]
      train:
        batch_size: 1
        steps: 1000
        gradient_accumulation_steps: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        ema_config:
          use_ema: true
          ema_decay: 0.99
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        is_flux: true
        quantize: true
"""

with open("/content/ai-toolkit/config_marcos.yaml", "w") as f:
    f.write(config_yaml)

print("\n Setup concluído! Para iniciar o treinamento de alta fidelidade, execute a célula abaixo com:")
print("!python run.py config_marcos.yaml")
