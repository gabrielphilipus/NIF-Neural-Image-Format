import torch
import os
import argparse
import numpy as np

def inspect_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Erro: Checkpoint não encontrado em '{checkpoint_path}'")
        return
        
    print("=================================================================================")
    print(f" RELATÓRIO DE INSPEÇÃO DE PESOS: {checkpoint_path}")
    print("=================================================================================")
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"Erro ao carregar o checkpoint: {e}")
        return
        
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)) if isinstance(checkpoint, dict) else checkpoint
    
    keys = list(state_dict.keys())
    print(f"Total de chaves de parâmetros no State Dict: {len(keys)}")
    
    # Padrões de interesse
    patterns = ["cond_enc", "cond_dec", "latent_scaler", "importance_network"]
    
    print("\nEstatísticas de pesos dos blocos de modulação/condicionamento:")
    print("-" * 97)
    print(f"{'Nome do Parâmetro (Layer)':<56} | {'Média':^10} | {'Desv. Padrão':^12} | {'Mín':^6} | {'Máx':^6}")
    print("-" * 97)
    
    for pattern in patterns:
        matching_keys = [k for k in keys if pattern in k and "weight" in k]
        if matching_keys:
            # Seleciona a última camada de pesos de modulação de cada bloco para auditoria
            target_key = matching_keys[-1]
            weights = state_dict[target_key].numpy()
            mean = np.mean(weights)
            std = np.std(weights)
            w_min = np.min(weights)
            w_max = np.max(weights)
            print(f"{target_key:<56} | {mean:10.6f} | {std:12.6f} | {w_min:6.2f} | {w_max:6.2f}")
        else:
            print(f"Padrão '{pattern:<15}'                                     | [Nenhum parâmetro correspondente]")
    print("-" * 97)
    
    # Diagnósticos científicos
    print("\nDIAGNÓSTICOS TÉCNICOS:")
    
    # 1. Latent Scaler
    scaler_keys = [k for k in keys if "latent_scaler" in k and "weight" in k]
    if scaler_keys:
        target_key = scaler_keys[-1]
        scaler_std = np.std(state_dict[target_key].numpy())
        print(f"\n* Modulador de Escala ({target_key}):")
        print(f"  - Desvio Padrão dos Pesos: {scaler_std:.5f}")
        if scaler_std > 0.12:
            print("  - [OK] Aprendizado Ativo: O Latent Scaler se moveu da inicialização e modula ativamente.")
        else:
            print("  - [WARNING] Baixa variação: O Latent Scaler está próximo da inicialização aleatória.")
            
    # 2. FiLM Encoder
    enc_keys = [k for k in keys if "cond_enc1" in k and "weight" in k]
    if enc_keys:
        target_key = enc_keys[-1]
        enc_std = np.std(state_dict[target_key].numpy())
        print(f"\n* Condicionador FiLM do Encoder ({target_key}):")
        print(f"  - Desvio Padrão dos Pesos: {enc_std:.5f}")
        if enc_std > 0.12:
            print("  - [OK] Aprendizado Ativo: MLP FiLM aprendeu modulações baseadas no embedding de q.")
        else:
            print("  - [WARNING] O condicionador FiLM apresenta desvio de pesos típico de inicialização aleatória (~0.08).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspeciona os desvios de pesos de checkpoints NIF.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_v3_production/nif_epoch_300.pth",
                        help="Caminho para o checkpoint .pth")
    args = parser.parse_args()
    
    inspect_checkpoint(args.checkpoint)
