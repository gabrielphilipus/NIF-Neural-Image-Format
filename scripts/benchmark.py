import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import json
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


def benchmark_traditional(img, format_name, quality_list):
    """
    Compacta e descompacta a imagem usando formatos tradicionais do Pillow (JPEG, WebP, AVIF).
    Retorna o bpp e as imagens reconstruídas correspondentes.
    """
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


def plot_curves(all_data, metric_name, output_path, is_aggregated=False):
    """
    Gera o gráfico de curvas Rate-Distortion (Bpp vs Métrica).
    """
    plt.figure(figsize=(10, 6))
    
    for format_name, points in all_data.items():
        if not points:
            continue
        
        if is_aggregated:
            # Ordena os pontos pelo bpp_mean (eixo x)
            points = sorted(points, key=lambda p: p["bpp_mean"])
            bpps = [p["bpp_mean"] for p in points]
            metrics = [p[f"{metric_name}_mean"] for p in points]
            bpp_stds = [p["bpp_std"] for p in points]
            metric_stds = [p[f"{metric_name}_std"] for p in points]
            
            plt.errorbar(bpps, metrics, xerr=bpp_stds, yerr=metric_stds, marker='o', label=format_name, linewidth=2, capsize=4)
        else:
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
    parser.add_argument("--image", type=str, required=True, help="Imagem ou diretório de teste para o benchmark")
    parser.add_argument("--output_dir", type=str, default="results", help="Pasta para salvar os gráficos e dados")
    parser.add_argument("--no_lpips", action="store_true", help="Desativar métrica perceptual LPIPS")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executando Benchmark em: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    # Identificar se é uma imagem ou diretório
    if os.path.isfile(args.image):
        images_to_eval = [args.image]
        is_directory = False
    elif os.path.isdir(args.image):
        images_to_eval = [os.path.join(args.image, f) for f in os.listdir(args.image) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        is_directory = True
    else:
        print(f"Caminho não encontrado: {args.image}")
        return

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

    if not is_directory:
        # Modo de imagem única (compatibilidade)
        img_path = images_to_eval[0]
        print(f"Processando imagem única: {img_path}")
        print("Processando compressões NIF...")
        nif_raw, img_aligned = benchmark_nif(model, img_path, nif_qualities, device)

        print("Processando compressões tradicionais...")
        jpeg_raw = benchmark_traditional(img_aligned, "JPEG", traditional_qualities)
        webp_raw = benchmark_traditional(img_aligned, "WEBP", traditional_qualities)
        avif_raw = benchmark_traditional(img_aligned, "AVIF", traditional_qualities) if has_avif else []

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
            psnr, msssim, lp_val = calculate_metrics(img_aligned, pt["img_rec"], device, ms_ssim_metric, lpips_metric)
            formats_data["JPEG"].append({"bpp": pt["bpp"], "psnr": psnr, "ms_ssim": msssim, "lpips": lp_val})

        # Métricas WebP
        for pt in webp_raw:
            psnr, msssim, lp_val = calculate_metrics(img_aligned, pt["img_rec"], device, ms_ssim_metric, lpips_metric)
            formats_data["WebP"].append({"bpp": pt["bpp"], "psnr": psnr, "ms_ssim": msssim, "lpips": lp_val})

        # Métricas AVIF
        for pt in avif_raw:
            psnr, msssim, lp_val = calculate_metrics(img_aligned, pt["img_rec"], device, ms_ssim_metric, lpips_metric)
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

    else:
        # Modo de diretório/lote
        print(f"Iniciando benchmark em lote para {len(images_to_eval)} imagens...")
        raw_results = {
            "JPEG": {},
            "WebP": {},
            "NIF": {}
        }
        if has_avif:
            raw_results["AVIF"] = {}

        for i_img, img_path in enumerate(images_to_eval):
            img_name = os.path.basename(img_path)
            print(f"[{i_img + 1}/{len(images_to_eval)}] Processando {img_name}...")
            
            try:
                # NIF
                nif_raw, img_aligned = benchmark_nif(model, img_path, nif_qualities, device)
                
                # Tradicionais
                jpeg_raw = benchmark_traditional(img_aligned, "JPEG", traditional_qualities)
                webp_raw = benchmark_traditional(img_aligned, "WEBP", traditional_qualities)
                avif_raw = benchmark_traditional(img_aligned, "AVIF", traditional_qualities) if has_avif else []
                
                # Métricas JPEG
                raw_results["JPEG"][img_name] = []
                for pt in jpeg_raw:
                    psnr, msssim, lp_val = calculate_metrics(img_aligned, pt["img_rec"], device, ms_ssim_metric, lpips_metric)
                    raw_results["JPEG"][img_name].append({"bpp": pt["bpp"], "psnr": psnr, "ms_ssim": msssim, "lpips": lp_val})
                
                # Métricas WebP
                raw_results["WebP"][img_name] = []
                for pt in webp_raw:
                    psnr, msssim, lp_val = calculate_metrics(img_aligned, pt["img_rec"], device, ms_ssim_metric, lpips_metric)
                    raw_results["WebP"][img_name].append({"bpp": pt["bpp"], "psnr": psnr, "ms_ssim": msssim, "lpips": lp_val})
                
                # Métricas AVIF
                if has_avif:
                    raw_results["AVIF"][img_name] = []
                    for pt in avif_raw:
                        psnr, msssim, lp_val = calculate_metrics(img_aligned, pt["img_rec"], device, ms_ssim_metric, lpips_metric)
                        raw_results["AVIF"][img_name].append({"bpp": pt["bpp"], "psnr": psnr, "ms_ssim": msssim, "lpips": lp_val})
                
                # Métricas NIF
                raw_results["NIF"][img_name] = []
                for pt in nif_raw:
                    psnr, msssim, lp_val = calculate_metrics(img_aligned, pt["img_rec"], device, ms_ssim_metric, lpips_metric)
                    raw_results["NIF"][img_name].append({"bpp": pt["bpp"], "psnr": psnr, "ms_ssim": msssim, "lpips": lp_val})
            except Exception as e:
                print(f"Erro ao processar {img_name}: {e}")

        # Agregação dos resultados
        print("Agregando resultados do dataset...")
        aggregated_results = {
            "JPEG": [],
            "WebP": [],
            "NIF": []
        }
        if has_avif:
            aggregated_results["AVIF"] = []

        # Para formatos e seus respectivos quality_lists/nominal qualities
        formats_config = [
            ("JPEG", traditional_qualities),
            ("WebP", traditional_qualities),
            ("NIF", nif_qualities)
        ]
        if has_avif:
            formats_config.append(("AVIF", traditional_qualities))

        for fmt, q_list in formats_config:
            # Temos len(q_list) qualidades nominais
            for idx, q_nominal in enumerate(q_list):
                bpps = []
                psnrs = []
                msssims = []
                lpipss = []
                
                for img_name, pts in raw_results[fmt].items():
                    if idx < len(pts):
                        bpps.append(pts[idx]["bpp"])
                        psnrs.append(pts[idx]["psnr"])
                        msssims.append(pts[idx]["ms_ssim"])
                        lpipss.append(pts[idx]["lpips"])
                
                if bpps:
                    bpp_mean, bpp_std = np.mean(bpps), np.std(bpps)
                    psnr_mean, psnr_std = np.mean(psnrs), np.std(psnrs)
                    ms_ssim_mean, ms_ssim_std = np.mean(msssims), np.std(msssims)
                    lpips_mean, lpips_std = np.mean(lpipss), np.std(lpipss)
                    
                    aggregated_results[fmt].append({
                        "quality": q_nominal,
                        "bpp_mean": float(bpp_mean),
                        "bpp_std": float(bpp_std),
                        "psnr_mean": float(psnr_mean),
                        "psnr_std": float(psnr_std),
                        "ms_ssim_mean": float(ms_ssim_mean),
                        "ms_ssim_std": float(ms_ssim_std),
                        "lpips_mean": float(lpips_mean),
                        "lpips_std": float(lpips_std)
                    })

        # Salva benchmark_aggregated.json
        save_data = {
            "raw_results": raw_results,
            "aggregated_results": aggregated_results
        }
        json_path = os.path.join(args.output_dir, "benchmark_aggregated.json")
        with open(json_path, "w") as f:
            json.dump(save_data, f, indent=4)
        print(f"Resultados agregados salvos em: {json_path}")

        # Desenha gráficos
        plot_curves(aggregated_results, "psnr", os.path.join(args.output_dir, "rd_curve_psnr.png"), is_aggregated=True)
        plot_curves(aggregated_results, "ms_ssim", os.path.join(args.output_dir, "rd_curve_msssim.png"), is_aggregated=True)
        if not args.no_lpips:
            plot_curves(aggregated_results, "lpips", os.path.join(args.output_dir, "rd_curve_lpips.png"), is_aggregated=True)

    print("\n" + "="*50)
    print(" BENCHMARK CONCLUÍDO COM SUCESSO!")
    print("="*50)
    print(f"Os gráficos de comparação foram salvos na pasta '{args.output_dir}'.")
    print(f"Formatos avaliados: JPEG, WebP{', AVIF' if has_avif else ''}, NIF")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
