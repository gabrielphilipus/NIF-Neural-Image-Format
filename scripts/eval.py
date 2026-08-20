import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_msssim import MS_SSIM
import lpips

from src.models.nif_codec import NIFCodec

# Utilitário para cálculo de PSNR
def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


@torch.no_grad()
def evaluate_image(model, image_path, q_val, device, lpips_metric, ms_ssim_metric):
    """
    Avalia o modelo NIF em uma única imagem para um determinado fator de qualidade 'q'.
    Retorna o bpp, psnr, ms-ssim, lpips e a imagem reconstruída.
    """
    img = Image.open(image_path).convert("RGB")
    
    # Prepara a imagem (assegura que largura/altura sejam divisíveis por 64 para evitar erros do autoencoder)
    w, h = img.size
    w_new = (w // 64) * 64
    h_new = (h // 64) * 64
    if w != w_new or h != h_new:
        # Usamos crop central para não distorcer a imagem
        transform = transforms.Compose([
            transforms.CenterCrop((h_new, w_new)),
            transforms.ToTensor()
        ])
    else:
        transform = transforms.ToTensor()
        
    x = transform(img).unsqueeze(0).to(device)
    B, C, H, W = x.size()
    num_pixels = B * H * W

    # Cria o tensor de qualidade condicional
    quality = torch.tensor([[q_val]], device=device)

    # Executa o modelo em modo eval
    out = model(x, quality)
    
    x_hat = torch.clamp(out["x_hat"], 0.0, 1.0)
    likelihoods = out["likelihoods"]

    # Cálculo do bpp
    bpp_y = torch.log(likelihoods["y"]).sum() / (-num_pixels * torch.log(torch.tensor(2.0, device=device)))
    bpp_z = torch.log(likelihoods["z"]).sum() / (-num_pixels * torch.log(torch.tensor(2.0, device=device)))
    bpp = (bpp_y + bpp_z).item()

    # Cálculo das métricas de distorção
    psnr = compute_psnr(x, x_hat)
    msssim = ms_ssim_metric(x, x_hat).item()
    
    if lpips_metric is not None:
        lp_val = lpips_metric(x, x_hat).mean().item()
    else:
        lp_val = 0.0

    # Reconverte x_hat para imagem PIL
    x_hat_np = x_hat.squeeze(0).cpu().permute(1, 2, 0).numpy()
    x_hat_np = (x_hat_np * 255.0).astype(np.uint8)
    reconstructed_img = Image.fromarray(x_hat_np)

    return bpp, psnr, msssim, lp_val, reconstructed_img


def main():
    parser = argparse.ArgumentParser(description="Script de Avaliação e Teste NIF")
    parser.add_argument("--checkpoint", type=str, required=True, help="Caminho para o checkpoint .pth do modelo")
    parser.add_argument("--image", type=str, required=True, help="Caminho da imagem ou diretório de teste")
    parser.add_argument("--q_steps", type=str, default="0.1,0.3,0.5,0.7,0.9", help="Passos de qualidade (q) separados por vírgula")
    parser.add_argument("--save_output", action="store_true", help="Salvar as imagens reconstruídas em disco")
    parser.add_argument("--output_dir", type=str, default="results", help="Diretório para salvar as saídas")
    parser.add_argument("--no_lpips", action="store_true", help="Desativar cálculo da métrica LPIPS")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Executando avaliação em: {device}")

    # Carrega o modelo
    model = NIFCodec(num_filters=128, latent_dim=192, num_slices=8).to(device)
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Checkpoint carregado com sucesso da época {checkpoint['epoch']}")
    except Exception as e:
        print(f"Erro ao carregar o checkpoint: {e}")
        return

    # Inicializa métricas de validação
    ms_ssim_metric = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    if not args.no_lpips:
        lpips_metric = lpips.LPIPS(net='alex').to(device)
        lpips_metric.eval()
    else:
        lpips_metric = None

    # Parsing dos passos de qualidade
    qs = [float(q) for q in args.q_steps.split(",")]

    # Identificar se é uma imagem ou diretório
    if os.path.isfile(args.image):
        images_to_eval = [args.image]
    elif os.path.isdir(args.image):
        images_to_eval = [os.path.join(args.image, f) for f in os.listdir(args.image) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    else:
        print(f"Caminho não encontrado: {args.image}")
        return

    print(f"Avaliando {len(images_to_eval)} imagem(ns)...")

    if args.save_output:
        os.makedirs(args.output_dir, exist_ok=True)

    # Imprimir Cabeçalho dos Resultados
    print("\n" + "="*80)
    print(f"{'Imagem':<20} | {'Qualidade q':<11} | {'Bpp (v)':<10} | {'PSNR (^)':<10} | {'MS-SSIM (^)':<12} | {'LPIPS (v)':<10}")
    print("="*80)

    for img_path in images_to_eval:
        img_name = os.path.basename(img_path)
        for q in qs:
            try:
                bpp, psnr, msssim, lp_val, rec_img = evaluate_image(
                    model, img_path, q, device, lpips_metric, ms_ssim_metric
                )
                
                # Imprime linha de resultado
                print(f"{img_name[:20]:<20} | {q:<11.2f} | {bpp:<10.4f} | {psnr:<10.2f} | {msssim:<12.5f} | {lp_val:<10.5f}")
                
                if args.save_output:
                    # Salva imagem reconstruída
                    base_name, ext = os.path.splitext(img_name)
                    save_name = f"{base_name}_q{q:.2f}{ext}"
                    rec_img.save(os.path.join(args.output_dir, save_name))
            except Exception as e:
                print(f"Erro ao avaliar {img_name} no q={q}: {e}")
                
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
