import os
import sys
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

print("[LOCAL INSIGHTFACE] Iniciando Motor de Troca Anatomica Local (InsightFace + GPU)...")

# 1. Carregar o modelo de visão e análise facial do InsightFace
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# 2. Carregar o modelo de swapper (inswapper_128)
swapper_model_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')
if not os.path.exists(swapper_model_path):
    print("[LOCAL INSIGHTFACE] Baixando modelo inswapper_128.onnx oficial do InsightFace...")
    swapper = insightface.model_zoo.get_model('inswapper_128.onnx', download=True, download_zip=True)
else:
    swapper = insightface.model_zoo.get_model(swapper_model_path, download=False, providers=['CPUExecutionProvider'])

def run_local_faceswap(body_img_path, face_img_path, output_path):
    print(f"[LOCAL INSIGHTFACE] Foto 1 (Corpo/Cenario Mestre): {body_img_path}")
    print(f"[LOCAL INSIGHTFACE] Foto 2 (Fonte do Rosto HD): {face_img_path}")
    
    img_body = cv2.imread(body_img_path)
    img_face = cv2.imread(face_img_path)
    
    if img_body is None or img_face is None:
        raise ValueError("Não foi possível carregar uma das imagens de entrada.")
        
    # Extrair faces
    faces_body = app.get(img_body)
    faces_face = app.get(img_face)
    
    if len(faces_body) == 0:
        raise ValueError("Nenhum rosto encontrado na foto do corpo.")
    if len(faces_face) == 0:
        raise ValueError("Nenhum rosto encontrado na foto HD de referência.")
        
    target_face = faces_body[0]
    source_face = faces_face[0]
    
    print("[LOCAL INSIGHTFACE] Efetuando transplante anatomico 3D de rosto sem transparencias...")
    res = swapper.get(img_body, target_face, source_face, paste_back=True)
    
    cv2.imwrite(output_path, res)
    print(f"[LOCAL INSIGHTFACE] Transplante Concluido com Sucesso! Imagem salva em: {output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) > 3:
        run_local_faceswap(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Uso: python local_faceswap_engine.py <foto_corpo> <foto_rosto_hd> <saida>")
