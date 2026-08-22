import subprocess
import os
import sys

def run_sweep(config_name, w_mse, w_ssim, w_lpips, save_dir):
    print("\n" + "#"*80)
    print(f" EXECUTANDO {config_name}: w_mse={w_mse}, w_ssim={w_ssim}, w_lpips={w_lpips}")
    print("#"*80 + "\n")
    
    cmd = [
        sys.executable, "scripts/sweep_fine_tune.py",
        "--config_name", config_name,
        "--w_mse", str(w_mse),
        "--w_ssim", str(w_ssim),
        "--w_lpips", str(w_lpips),
        "--lambda_min", "0.15",
        "--lambda_max", "15.0",
        "--epochs", "15",
        "--batch_size", "8",
        "--lr", "1e-4",
        "--dataset", "DIV2K_train_HR/",
        "--val_dataset", "kodak24/",
        "--base_checkpoint", "checkpoints_v4_production/nif_epoch_300.pth",
        "--save_dir", save_dir
    ]
    
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[ERRO] Falha na execução de {config_name}")
    else:
        print(f"[SUCESSO] {config_name} concluído!")

def main():
    print("=== INICIANDO SWEEP DE CALIBRAÇÃO (CONFIGS A, B, C) ===")
    
    # 1. Config A (Balanced)
    run_sweep("Config_A_Balanced", 0.40, 0.40, 0.20, "checkpoints_sweep_A")
    
    # 2. Config B (High-Fidelity)
    run_sweep("Config_B_HighFidelity", 0.65, 0.25, 0.10, "checkpoints_sweep_B")
    
    # 3. Config C (Pure-Fidelity)
    run_sweep("Config_C_PureFidelity", 0.85, 0.15, 0.00, "checkpoints_sweep_C")
    
    print("\n" + "="*80)
    print(" TODOS OS 3 SWEEPS FORAM CONCLUÍDOS!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
