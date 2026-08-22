import sys
import os
sys.path.append(os.getcwd())
import subprocess

def main():
    print("=== EXPERIMENTO DE DESAMBIGUAÇÃO: TETO DE CAPACIDADE VS. UNDER-TRAINING ===")
    print("Executando Config D (Pure-Fidelity Extendido: 50 Épocas, GAN=0, w_mse=0.90, w_ssim=0.10, w_lpips=0.0)...")
    
    cmd = [
        sys.executable, "scripts/sweep_fine_tune.py",
        "--config_name", "Config_D_Disambiguation_50ep",
        "--w_mse", "0.90",
        "--w_ssim", "0.10",
        "--w_lpips", "0.00",
        "--lambda_min", "0.15",
        "--lambda_max", "15.0",
        "--epochs", "50",
        "--batch_size", "8",
        "--lr", "1.5e-4",
        "--dataset", "DIV2K_train_HR/",
        "--val_dataset", "kodak24/",
        "--base_checkpoint", "checkpoints_v4_production/nif_epoch_300.pth",
        "--save_dir", "checkpoints_disambiguation_D"
    ]
    
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n[SUCESSO] Experimento de desambiguação concluído!")
    else:
        print(f"\n[ERRO] Falha no experimento de desambiguação (código {res.returncode})")

if __name__ == "__main__":
    main()
