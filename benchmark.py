import argparse
import json
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from pytorch_msssim import MS_SSIM
import lpips
import matplotlib.pyplot as plt

# Tenta importar suporte a AVIF do pillow_heif
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    has_avif = True
except ImportError:
    has_avif = False

from src.models.nif_codec import NIFCodec
from eval import compute_psnr


def benchmark_traditional(img_path, format_name, quality_list):
    """
    Compacta e descompacta a imagem usando formatos tradicionais do Pillow (JPEG, WebP, AVIF).
    Retorna o bpp e as imagens reconstruídas correspondentes.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    num_pixels = w * h
    results = []

    for q in quality_list:
        ext = "jpg" if format_name == "JPEG" else format_name.lower()
        temp_file = f"temp_bench_{format_name.lower()}_{q}.{ext}"
        
        try:
            if format_name == "JPEG":
                img.save(temp_file, format="JPEG", quality=q)
            elif format_name == "WEBP":
                img.save(temp_file, format="WEBP", quality=q)
            elif format_name == "AVIF":
                if not has_avif:
                    continue
                img.save(temp_file, format="AVIF", quality=q)
                
            # Calcula tamanho em bytes
            file_size = os.path.getsize(temp_file)
            bpp = (file_size * 8) / num_pixels
            
            # Carrega de volta
            img_rec = Image.open(temp_file).convert("RGB")
            
            results.append({
                "bpp": bpp,
                "img_rec": img_rec
            })
        except Exception as e:
            print(f"Erro ao codificar em {format_name} com qualidade {q}: {e}")
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    return results


@torch.no_grad()
def benchmark_nif(model, img_path, quality_list, device):
    """
    Roda a simulação de compressão do modelo NIF para uma lista de qualidades 'q'.
    Retorna os bpp e as imagens reconstruídas.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    
    # Recorta para múltiplo de 64
    w_new = (w // 64) * 64
    h_new = (h // 64) * 64
    if w != w_new or h != h_new:
        transform = transforms.CenterCrop((h_new, w_new))
        img = transform(img)
        w, h = w_new, h_new
        
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    num_pixels = w * h
    results = []

    for q_val in quality_list:
        quality = torch.tensor([[q_val]], device=device)
        out = model(x, quality)
        
        x_hat = torch.clamp(out["x_hat"], 0.0, 1.0)
        likelihoods = out["likelihoods"]

        # Calcula o bpp
        bpp_y = torch.log(likelihoods["y"]).sum() / (-num_pixels * torch.log(torch.tensor(2.0, device=device)))
        bpp_z = torch.log(likelihoods["z"]).sum() / (-num_pixels * torch.log(torch.tensor(2.0, device=device)))
        bpp = (bpp_y + bpp_z).item()

        # Recria a imagem PIL
        x_hat_np = x_hat.squeeze(0).cpu().permute(1, 2, 0).numpy()
        x_hat_np = (x_hat_np * 255.0).astype(np.uint8)
        img_rec = Image.fromarray(x_hat_np)
        
        results.append({
            "bpp": bpp,
            "img_rec": img_rec
        })
        
    return results, img


def calculate_metrics(img_orig, img_rec, device, ms_ssim_metric, lpips_metric):
    """
    Calcula PSNR, MS-SSIM e LPIPS entre a imagem original e a reconstruída.
    """
    transform = transforms.ToTensor()
    x = transform(img_orig).unsqueeze(0).to(device)
    x_hat = transform(img_rec).unsqueeze(0).to(device)
    
    psnr = compute_psnr(x, x_hat)
    msssim = ms_ssim_metric(x, x_hat).item()
    
    if lpips_metric is not None:
        lp_val = lpips_metric(x, x_hat).mean().item()
    else:
        lp_val = 0.0
        
    return psnr, msssim, lp_val


def plot_curves(all_data, metric_name, output_path):
    """
    Gera o gráfico de curvas Rate-Distortion (Bpp vs Métrica).
    """
    plt.figure(figsize=(10, 6))
    
    for format_name, points in all_data.items():
        if not points:
            continue
        # Ordena os pontos pelo bpp (eixo x)
        points = sorted(points, key=lambda p: p["bpp"])
        bpps = [p["bpp"] for p in points]
        metrics = [p[metric_name] for p in points]
        
        plt.plot(bpps, metrics, marker='o', label=format_name, linewidth=2)
        
    plt.xlabel("Bitrate (Bits per Pixel - Bpp)", fontsize=12)
    
    if metric_name == "psnr":
        plt.ylabel("PSNR (dB - Quanto maior, melhor)", fontsize=12)
        plt.title("Curva Rate-Distortion: PSNR vs Bitrate", fontsize=14, fontweight="bold")
    elif metric_name == "ms_ssim":
        plt.ylabel("MS-SSIM (Quanto maior, melhor)", fontsize=12)
        plt.title("Curva Rate-Distortion: MS-SSIM vs Bitrate", fontsize=14, fontweight="bold")
    elif metric_name == "lpips":
        plt.ylabel("LPIPS (Quanto menor, melhor)", fontsize=12)
        plt.title("Curva Rate-Distortion: LPIPS vs Bitrate", fontsize=14, fontweight="bold")
        
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Gráfico salvo em: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Script de Benchmark Comparativo NIF vs JPEG vs WebP vs AVIF")
    parser.add_argument("--checkpoint", type=str, default="", help="Caminho do checkpoint do NIF (Opcional, inicializa pesos aleatórios para teste se vazio)")
    parser.add_argument("--image", type=str, required=True, help="Imagem de teste para o benchmark")
    parser.add_argument("--output_dir", type=str, default="results", help="Pasta para salvar os gráficos e dados")
    parser.add_argument("--no_lpips", action="store_true", help="Desativar métrica perceptual LPIPS")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executando Benchmark em: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Carrega modelo NIF
    model = NIFCodec(num_filters=128, latent_dim=192, num_slices=8).to(device)
    if args.checkpoint:
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Pesos carregados com sucesso do checkpoint: {args.checkpoint}")
        except Exception as e:
            print(f"Erro ao carregar o checkpoint: {e}")
            return
    else:
        print("[Aviso] Sem checkpoint. Rodando com pesos aleatórios para teste de fluxo.")
    model.eval()

    # Inicializa métricas
    ms_ssim_metric = MS_SSIM(data_range=1.0, size_average=True, channel=3).to(device)
    if not args.no_lpips:
        lpips_metric = lpips.LPIPS(net='alex').to(device)
        lpips_metric.eval()
    else:
        lpips_metric = None

    # 2. Executa compressões
    traditional_qualities = [10, 25, 50, 75, 90]
    nif_qualities = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Obter os resultados brutos (Bpp e imagens de saída)
    print("Processando compressões tradicionais...")
    jpeg_raw = benchmark_traditional(args.image, "JPEG", traditional_qualities)
    webp_raw = benchmark_traditional(args.image, "WEBP", traditional_qualities)
    avif_raw = benchmark_traditional(args.image, "AVIF", traditional_qualities) if has_avif else []
    
    print("Processando compressões NIF...")
    nif_raw, img_aligned = benchmark_nif(model, args.image, nif_qualities, device)

    # 3. Calcula as métricas quantitativas comparando com img_aligned
    # (Todas as métricas são calculadas usando a imagem com as dimensões alinhadas do NIF para comparação justa)
    print("Calculando métricas para cada ponto de qualidade...")
    formats_data = {
        "JPEG": [],
        "WebP": [],
        "NIF": []
    }
    if has_avif:
        formats_data["AVIF"] = []

    # Métricas JPEG
    for pt in jpeg_raw:
        # Re-recorta a imagem do JPEG para alinhar com o corte do NIF para fins de comparação
        img_rec_aligned = pt["img_rec"].crop((0, 0, img_aligned.size[0], img_aligned.size[1]))
        psnr, msssim, lp_val = calculate_metrics(img_aligned, img_rec_aligned, device, ms_ssim_metric, lpips_metric)
        formats_data["JPEG"].append({"bpp": pt["bpp"], "psnr": psnr, "ms_ssim": msssim, "lpips": lp_val})

    # Métricas WebP
    for pt in webp_raw:
        img_rec_aligned = pt["img_rec"].crop((0, 0, img_aligned.size[0], img_aligned.size[1]))
        psnr, msssim, lp_val = calculate_metrics(img_aligned, img_rec_aligned, device, ms_ssim_metric, lpips_metric)
        formats_data["WebP"].append({"bpp": pt["bpp"], "psnr": psnr, "ms_ssim": msssim, "lpips": lp_val})

    # Métricas AVIF
    for pt in avif_raw:
        img_rec_aligned = pt["img_rec"].crop((0, 0, img_aligned.size[0], img_aligned.size[1]))
        psnr, msssim, lp_val = calculate_metrics(img_aligned, img_rec_aligned, device, ms_ssim_metric, lpips_metric)
        formats_data["AVIF"].append({"bpp": pt["bpp"], "psnr": psnr, "ms_ssim": msssim, "lpips": lp_val})

    # Métricas NIF
    for pt in nif_raw:
        psnr, msssim, lp_val = calculate_metrics(img_aligned, pt["img_rec"], device, ms_ssim_metric, lpips_metric)
        formats_data["NIF"].append({"bpp": pt["bpp"], "psnr": psnr, "ms_ssim": msssim, "lpips": lp_val})

    # 4. Salva dados brutos em JSON
    json_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(formats_data, f, indent=4)
    print(f"Dados salvos em: {json_path}")

    # 5. Desenha e salva os gráficos
    plot_curves(formats_data, "psnr", os.path.join(args.output_dir, "rd_curve_psnr.png"))
    plot_curves(formats_data, "ms_ssim", os.path.join(args.output_dir, "rd_curve_msssim.png"))
    if not args.no_lpips:
        plot_curves(formats_data, "lpips", os.path.join(args.output_dir, "rd_curve_lpips.png"))

    print("\n" + "="*50)
    print(" BENCHMARK CONCLUÍDO COM SUCESSO!")
    print("="*50)
    print(f"Os gráficos de comparação foram salvos na pasta '{args.output_dir}'.")
    print(f"Formatos avaliados: JPEG, WebP{', AVIF' if has_avif else ''}, NIF")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
