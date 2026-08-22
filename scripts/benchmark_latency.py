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
    load_checkpoint = os.path.exists(checkpoint_path)
    
    print("="*85)
    print(" NIF BENCHMARK - LATÊNCIA, CUSTO OPERACIONAL (AWS) & PARALELISMO DE CONTEXTO")
    print("="*85)

    if not load_checkpoint:
        print("Aviso: Checkpoint 'checkpoints/nif_epoch_300.pth' não encontrado.")
        print("Rodando benchmark de latência física com pesos inicializados aleatoriamente (latência idêntica).")
        print("="*85)

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
        
        for slices in [8, 4, 2, 1]:
            print("\n" + "-" * 85)
            print(f"Configuração: {slices} Slices de Canais | {slices * 2} Passos Seriais de Decodificação")
            print("-" * 85)
            print(f"{'Resolução':<10} | {'Encode (ms)':<12} | {'Decode (ms)':<12} | {'Total (ms)':<10} | {'Custo/1M Imgs (USD)':<20}")
            print("-" * 85)

            # Load model on target device
            model = NIFCodec(num_filters=128, latent_dim=192, num_slices=slices).to(dev)
            if load_checkpoint and slices == 8:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location=dev)
                    model.load_state_dict(checkpoint['model_state_dict'])
                except Exception as e:
                    print(f"Erro ao carregar o checkpoint: {e}")
            
            model.update(force=True)
            model.eval()

            for res in resolutions:
                # O latente do NIFCodec exige resoluções múltiplas de 64
                x = torch.randn(1, 3, res, res, device=dev)
                quality = torch.tensor([[0.5]], device=dev)

                try:
                    enc_t, dec_t = measure_time(model, x, quality, dev)
                    total_t = enc_t + dec_t
                    cost_1m = (total_t / 3600000.0) * 1000000.0 * aws_hourly_rate
                    print(f"{f'{res}x{res}':<10} | {enc_t:<12.2f} | {dec_t:<12.2f} | {total_t:<10.2f} | ${cost_1m:<19.4f}")
                except Exception as e:
                    print(f"{f'{res}x{res}':<10} | Erro: {e}")
            print("-" * 85)

if __name__ == "__main__":
    main()
