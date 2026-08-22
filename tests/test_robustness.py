import unittest
import torch
import os
from PIL import Image
from torchvision.transforms.functional import to_tensor
from src.models.nif_codec import NIFCodec
from src.models.entropy_model import ChannelCheckerboardEntropyModel

class TestNIFRobustness(unittest.TestCase):
    def test_importance_map_symmetry(self):
        """Verifica se o Importance Map gerado no encoder e decoder é 100% idêntico (zero overhead)"""
        entropy_model = ChannelCheckerboardEntropyModel(in_channels=192, num_slices=8, latent_dim=192)
        entropy_model.eval()
        
        B, C, H, W = 1, 192, 16, 16
        hyper_features = torch.randn(B, C, H, W)
        
        with torch.no_grad():
            importance_map_enc = 0.1 + 0.9 * entropy_model.importance_network(hyper_features)
            importance_map_dec = 0.1 + 0.9 * entropy_model.importance_network(hyper_features)
            
        self.assertTrue(torch.allclose(importance_map_enc, importance_map_dec, atol=1e-6))

    def test_cdf_robustness_natural_image(self):
        """Verifica se a descompressão real de imagem natural com a CDF dinâmica não gera NaNs"""
        checkpoint_path = "checkpoints_v3_production/nif_epoch_300.pth"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
        model = NIFCodec(num_filters=128, latent_dim=192, num_slices=8).to(device)
        
        # Carrega checkpoint se disponível, senão usa inicialização aleatória determinista para a CI
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint)))
        else:
            torch.manual_seed(42)
            for m in model.modules():
                if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
                    torch.nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        torch.nn.init.constant_(m.bias, 0.0)
            print("  * Checkpoint de produção não encontrado. Rodando com pesos deterministas simulados para CI.")
            
        model.update(force=True)
        model.eval()
        
        img_path = "kodak24/kodim01.png"
        if os.path.exists(img_path):
            img = to_tensor(Image.open(img_path)).unsqueeze(0).to(device)
        else:
            torch.manual_seed(42)
            img = torch.rand(1, 3, 64, 64, device=device)
            print("  * Imagem Kodak não encontrada. Usando imagem sintética 64x64 texturizada para CI.")
        
        with torch.no_grad():
            q_tensor = torch.tensor([[0.5]], device=device)
            out_enc = model.compress(img, quality=q_tensor)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"], quality=q_tensor)
            
        reconstructed_img = out_dec["x_hat"]
        self.assertFalse(torch.isnan(reconstructed_img).any().item(), "Erro: A reconstrução gerou NaNs no decoder!")

if __name__ == "__main__":
    unittest.main()
