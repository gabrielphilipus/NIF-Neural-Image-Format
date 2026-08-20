import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import glob
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

from src.models.nif_codec import NIFCodec

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executando diagnóstico no dispositivo: {device}")
    
    # 1. Inicializa o modelo e carrega o checkpoint final
    model = NIFCodec(num_filters=128, latent_dim=192, num_slices=8).to(device)
    checkpoint_path = "checkpoints/nif_epoch_100.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Checkpoint '{checkpoint_path}' não encontrado.")
        return
        
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Checkpoint carregado com sucesso (Época {checkpoint['epoch']})")
    except Exception as e:
        print(f"Erro ao carregar o checkpoint: {e}")
        return

    # 2. Localiza a imagem da moça ou alguma imagem de teste no DIV2K
    # Procura arquivos contendo "mo" ou qualquer imagem na pasta
    img_files = glob.glob("DIV2K_train_HR/*mo*.png") + glob.glob("DIV2K_train_HR/*mo*.jpg")
    if not img_files:
        img_files = glob.glob("DIV2K_train_HR/*.png") + glob.glob("DIV2K_train_HR/*.jpg")
        
    if not img_files:
        print("Nenhuma imagem encontrada em 'DIV2K_train_HR' para testar.")
        return
        
    img_path = img_files[0]
    print(f"Imagem selecionada para o diagnóstico: {img_path}")
    
    # Prepara a imagem recortando para múltiplos de 64
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    w_new = (w // 64) * 64
    h_new = (h // 64) * 64
    img_cropped = transforms.CenterCrop((h_new, w_new))(img)
    
    x = transforms.ToTensor()(img_cropped).unsqueeze(0).to(device)
    quality = torch.tensor([[0.5]], device=device)
    
    # 3. Executa o Forward Pass direto (Simulação sem codificação aritmética)
    print("Executando simulação (Forward)...")
    with torch.no_grad():
        out_forward = model(x, quality)
        x_hat_forward = torch.clamp(out_forward["x_hat"], 0.0, 1.0)
        
    # Salva o resultado do Forward
    x_f_np = x_hat_forward.squeeze(0).cpu().permute(1, 2, 0).numpy()
    x_f_np = (x_f_np * 255.0).astype(np.uint8)
    Image.fromarray(x_f_np).save("diag_reconstruction_forward.png")
    print("Imagem simulada salva em: diag_reconstruction_forward.png")

    # 4. Executa a Compressão e Descompressão real (Codificação Aritmética)
    print("Executando compressão/descompressão real (Bitstream)...")
    try:
        model.update(force=True)
        with torch.no_grad():
            # Comprime
            compressed = model.compress(x, quality)
            # Descomprime
            decompressed = model.decompress(compressed["strings"], compressed["shape"], quality)
            x_hat_codec = torch.clamp(decompressed["x_hat"], 0.0, 1.0)
            
        # Salva o resultado do Codec
        x_c_np = x_hat_codec.squeeze(0).cpu().permute(1, 2, 0).numpy()
        x_c_np = (x_c_np * 255.0).astype(np.uint8)
        Image.fromarray(x_c_np).save("diag_reconstruction_codec.png")
        print("Imagem real do codec salva em: diag_reconstruction_codec.png")
        
        # Compara numericamente
        difference = torch.mean(torch.abs(x_hat_forward - x_hat_codec)).item()
        print(f"Diferença média absoluta entre Simulação e Codec: {difference:.6f}")
        if difference < 1e-4:
            print("SUCESSO: A simulação e a decodificação binária são idênticas!")
        else:
            print("ALERTA: Há discrepância entre a simulação e a decodificação binária.")
            
    except Exception as e:
        print(f"Erro no pipeline do codec: {e}")

if __name__ == "__main__":
    main()
