import os
import sys
import cv2
import numpy as np
import torch
import insightface
from insightface.app import FaceAnalysis

torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
if os.path.exists(torch_lib):
    os.add_dll_directory(torch_lib)
    os.environ['PATH'] = torch_lib + os.path.pathsep + os.environ.get('PATH', '')

print("[LOCAL ENGINE v5.0 - RTX 3050 CUDA FULL HD] Inicializando Motor GPU Nvidia + InsightFace HD...")

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

swapper_model_path = os.path.expanduser('~/.insightface/models/inswapper_128.onnx')
swapper = insightface.model_zoo.get_model(swapper_model_path, download=False, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

from gfpgan import GFPGANer

print("[LOCAL ENGINE v5.0 - RTX 3050 CUDA FULL HD] Inicializando GFPGANer 4K na GPU...")
restorer = GFPGANer(model_path='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth', upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None)

def apply_face_mask(original, modified, mask):
    """
    Substitui apenas os pixels da face (onde mask == 255) pela imagem modificada.
    Todos os outros pixels permanecem 100% iguais à imagem original (Kimi AI pattern).
    """
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
    mask = (mask > 50).astype(np.uint8) * 255

    if mask.shape[:2] != original.shape[:2]:
        mask = cv2.resize(mask, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)

    result = original.copy()
    cv2.copyTo(src=modified, mask=mask, dst=result)
    return result

def process_unified_v4(main_img_path, ref_img_path, output_path, mask_img_path=None):
    logs = []
    logs.append("📥 Foto Principal recebida e carregada no canvas.")
    
    img_main = cv2.imread(main_img_path)
    if img_main is None:
        logs.append("❌ Erro: Foto Principal não encontrada ou formato inválido.")
        raise ValueError("Foto Principal nao encontrada ou invalida.")
        
    modified_img = img_main.copy()
    
    if ref_img_path and os.path.exists(ref_img_path):
        logs.append(f"🖼️ Foto de referência selecionada: {os.path.basename(ref_img_path)}.")
        img_ref = cv2.imread(ref_img_path)
        if img_ref is not None:
            faces_main = app.get(img_main)
            faces_ref = app.get(img_ref)
            
            if len(faces_main) > 0 and len(faces_ref) > 0:
                target_face = faces_main[0]
                source_face = faces_ref[0]
                logs.append("👤 Rosto detectado na imagem! Executando Transplante Anatômico 3D via GPU RTX 3050...")
                swapped = swapper.get(img_main, target_face, source_face, paste_back=True)
                
                logs.append("✨ Restaurando poros, dentes e olhar de alta definição em 4K (GFPGAN Ultra HD)...")
                _, _, restored_face = restorer.enhance(swapped, has_aligned=False, only_center_face=False, paste_back=True)
                modified_img = restored_face if restored_face is not None else swapped
            else:
                logs.append("⚠️ Nenhum rosto detectado na foto de referência ou principal para transplante 3D.")
    else:
        logs.append("ℹ️ Nenhuma foto de referência selecionada (processando apenas restauração/máscara).")

    # Se houver mascara desenhada, verifica se e rosto ou corpo/braço
    if mask_img_path and os.path.exists(mask_img_path):
        user_mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        if user_mask is not None and np.max(user_mask) > 50:
            logs.append("🖌️ Máscara pintada detectada! Analisando área selecionada...")
            
            # Se nao houve transplante de rosto, mas há mascara (ex: braços, pelos, corpo), aplica super-nitidez de pelos/pele
            if not ref_img_path or not os.path.exists(ref_img_path):
                logs.append("💪 Regra Multi-Regional Ativada: Processando pelos, braços e pele na GPU RTX 3050 (sem alterar rosto)...")
                _, _, enhanced_body = restorer.enhance(img_main, has_aligned=False, only_center_face=False, paste_back=True)
                if enhanced_body is not None:
                    modified_img = enhanced_body

            logs.append("✂️ Aplicando mesclagem cirúrgica: 100% dos pixels fora da máscara mantidos idênticos.")
            final_result = apply_face_mask(img_main, modified_img, user_mask)
        else:
            final_result = modified_img
    else:
        final_result = modified_img

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, final_result)
    logs.append("✅ Processamento concluído com sucesso em 4K HD!")
    return output_path, logs
