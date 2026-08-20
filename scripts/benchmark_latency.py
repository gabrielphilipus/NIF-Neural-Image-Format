import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
from src.models.nif_codec import NIFCodec

def measure_time(model, x, quality, device, runs=10, warmup=3):
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            if device == "cuda":
                torch.cuda.synchronize()
            out_comp = model.compress(x, quality)
            out_dec = model.decompress(out_comp["strings"], out_comp["shape"], quality)
            if device == "cuda":
                torch.cuda.synchronize()

    # Measure Encode
    start_enc = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            if device == "cuda":
                torch.cuda.synchronize()
            out_comp = model.compress(x, quality)
            if device == "cuda":
                torch.cuda.synchronize()
    end_enc = time.perf_counter()
    avg_enc_time = (end_enc - start_enc) / runs * 1000  # ms

    # Measure Decode
    # Prepare compressed data
    with torch.no_grad():
        out_comp = model.compress(x, quality)
    
    start_dec = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            if device == "cuda":
                torch.cuda.synchronize()
            out_dec = model.decompress(out_comp["strings"], out_comp["shape"], quality)
            if device == "cuda":
                torch.cuda.synchronize()
    end_dec = time.perf_counter()
    avg_dec_time = (end_dec - start_dec) / runs * 1000  # ms

    return avg_enc_time, avg_dec_time

def main():
    checkpoint_path = 'checkpoints/nif_epoch_300.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Checkpoint '{checkpoint_path}' não encontrado.")
        return

    print("="*75)
    print(" NIF BENCHMARK - LATÊNCIA & CUSTO OPERACIONAL (AWS)")
    print("="*75)

    resolutions = [256, 512, 1024]
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        # Optimize backend
        torch.backends.cudnn.benchmark = True

    # g4dn.xlarge cost: $0.526 / hour
    aws_hourly_rate = 0.526

    for dev in devices:
        print(f"\nDispositivo de Execução: {dev.upper()}")
        print("-" * 75)
        print(f"{'Resolução':<10} | {'Encode (ms)':<12} | {'Decode (ms)':<12} | {'Total (ms)':<10} | {'Custo/1M Imgs (USD)':<20}")
        print("-" * 75)

        # Load model on target device
        model = NIFCodec(num_filters=128, latent_dim=192, num_slices=8).to(dev)
        checkpoint = torch.load(checkpoint_path, map_location=dev)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.update()
        model.eval()

        for res in resolutions:
            x = torch.randn(1, 3, res, res, device=dev)
            quality = torch.tensor([[0.5]], device=dev)

            try:
                enc_t, dec_t = measure_time(model, x, quality, dev)
                total_t = enc_t + dec_t
                
                # throughput = images per hour
                # throughput = (3600 seconds * 1000 ms) / total_t
                # cost per 1M images = (1,000,000 / throughput) * aws_hourly_rate
                # Cost formula simplified: (total_t / 3,600,000) * 1,000,000 * aws_hourly_rate
                cost_1m = (total_t / 3600000.0) * 1000000.0 * aws_hourly_rate

                print(f"{f'{res}x{res}':<10} | {enc_t:<12.2f} | {dec_t:<12.2f} | {total_t:<10.2f} | ${cost_1m:<19.4f}")
            except Exception as e:
                print(f"{f'{res}x{res}':<10} | Erro: {e}")
        print("-" * 75)

if __name__ == "__main__":
    main()
