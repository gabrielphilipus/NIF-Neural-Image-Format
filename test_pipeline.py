import unittest
import torch
import torch.nn as nn
from src.models.entropy_model import MaskedConv2d, ChannelCheckerboardEntropyModel
from src.models.nif_codec import NIFCodec
from train import NIFLoss

class TestNIFPipeline(unittest.TestCase):
    def setUp(self):
        # Configura dimensões padrão de teste
        self.batch_size = 2
        self.channels = 3
        self.height = 256
        self.width = 256
        self.latent_dim = 192
        self.num_slices = 8

    def test_masked_conv2d(self):
        """
        Verifica se o MaskedConv2d realmente mantém o peso central zerado.
        """
        conv = MaskedConv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        
        # Gera entrada dummy
        x = torch.randn(self.batch_size, 16, 32, 32)
        
        # Realiza forward pass
        out = conv(x)
        self.assertEqual(out.shape, x.shape)
        
        # Verifica se o peso central na máscara de pesos é zero
        kh, kw = conv.weight.shape[-2:]
        center_h, center_w = kh // 2, kw // 2
        
        # O peso mascarado resultante deve ter zero no centro
        masked_weight = conv.weight * conv.mask
        self.assertTrue(torch.all(masked_weight[:, :, center_h, center_w] == 0.0))

    def test_entropy_model_shapes(self):
        """
        Verifica as dimensões de saída do modelo de entropia híbrido.
        """
        entropy_model = ChannelCheckerboardEntropyModel(
            in_channels=self.latent_dim, 
            num_slices=self.num_slices, 
            latent_dim=self.latent_dim
        )
        
        # Dimensões do latente principal 'y'
        y = torch.randn(self.batch_size, self.latent_dim, 16, 16)
        
        # Dimensões da hyperprior reconstruída
        hyper_features = torch.randn(self.batch_size, self.latent_dim, 16, 16)
        
        y_hat, likelihoods = entropy_model(y, hyper_features)
        
        self.assertEqual(y_hat.shape, y.shape)
        self.assertEqual(likelihoods.shape, y.shape)
        self.assertTrue(torch.all(likelihoods >= 0.0))

    def test_nif_codec_forward(self):
        """
        Testa o fluxo completo do NIFCodec (Encoder -> Hyperprior -> Quantização -> Decoder).
        """
        model = NIFCodec(num_filters=64, latent_dim=128, num_slices=4)
        model.eval()
        
        x = torch.rand(self.batch_size, self.channels, self.height, self.width)
        quality = torch.rand(self.batch_size, 1)
        
        out = model(x, quality)
        
        self.assertIn("x_hat", out)
        self.assertIn("likelihoods", out)
        self.assertIn("sfm", out)
        
        x_hat = out["x_hat"]
        sfm = out["sfm"]
        likelihoods = out["likelihoods"]
        
        # A imagem reconstruída deve ter o mesmo tamanho da entrada
        self.assertEqual(x_hat.shape, x.shape)
        
        # A máscara SFM deve ter tamanho da entrada, mas canal único (escala de cinza)
        self.assertEqual(sfm.shape, (self.batch_size, 1, self.height, self.width))
        
        # Verificando as dimensões das probabilidades
        # y é reduzido espacialmente por um fator de 16 pelo encoder (4 camadas de stride 2)
        expected_y_h = self.height // 16
        expected_y_w = self.width // 16
        self.assertEqual(likelihoods["y"].shape, (self.batch_size, 128, expected_y_h, expected_y_w))
        
        # z (hyperprior) é reduzido espacialmente por mais um fator de 4 (totalizando 64)
        expected_z_h = self.height // 64
        expected_z_w = self.width // 64
        self.assertEqual(likelihoods["z"].shape, (self.batch_size, 64, expected_z_h, expected_z_w))

    def test_nif_loss_gradient(self):
        """
        Verifica se a função de perda computa corretamente e se os gradientes
        fluem por todo o modelo de ponta a ponta (verificação de diferenciabilidade).
        """
        model = NIFCodec(num_filters=64, latent_dim=128, num_slices=4)
        model.train()
        
        # Usamos use_lpips=False no teste unitário para evitar que faça download
        # automático do modelo AlexNet durante os testes locais automáticos do CI.
        criterion = NIFLoss(use_lpips=False)
        
        x = torch.rand(self.batch_size, self.channels, self.height, self.width, requires_grad=True)
        quality = torch.rand(self.batch_size, 1)
        
        out = model(x, quality)
        
        loss, rate, mse, msssim, _ = criterion(
            x, out["x_hat"], out["likelihoods"], out["sfm"], quality,
            lambda_min=0.001, lambda_max=0.04
        )
        
        # Verifica se o valor de loss é válido
        self.assertFalse(torch.isnan(loss))
        self.assertTrue(loss.item() > 0.0)
        
        # Executa backward
        loss.backward()
        
        # Verifica se os gradientes foram populados no codificador
        self.assertIsNotNone(model.enc_conv1.weight.grad)
        self.assertTrue(torch.any(model.enc_conv1.weight.grad != 0.0))

    def test_compress_decompress(self):
        """
        Verifica a codificação e decodificação aritmética real (bitstream).
        """
        model = NIFCodec(num_filters=64, latent_dim=128, num_slices=4)
        model.eval()
        
        # O modelo precisa inicializar as tabelas CDF de codificação aritmética
        model.update(force=True)
        
        # Codificação aritmética real é feita imagem por imagem (batch = 1)
        x = torch.rand(1, self.channels, self.height, self.width)
        quality = torch.tensor([[0.5]])
        
        # Comprime
        compressed_out = model.compress(x, quality)
        self.assertIn("strings", compressed_out)
        self.assertIn("shape", compressed_out)
        
        # Descomprime
        decompressed_out = model.decompress(
            compressed_out["strings"], 
            compressed_out["shape"], 
            quality
        )
        
        x_hat = decompressed_out["x_hat"]
        self.assertEqual(x_hat.shape, x.shape)

    def test_cli_tool_integration(self):
        """
        Verifica o fluxo completo de serialização e desserialização de arquivos binários .nif.
        """
        import os
        from nif_tool import save_nif_file, load_nif_file
        
        # Dados fictícios para simular saídas de compressão
        width, height = 256, 256
        quality_int = 127
        z_shape = (4, 4)
        z_string = [b"dummy_z_bytes"]
        y_strings = [[b"slice_0_anchor"], [b"slice_0_non_anchor"], [b"slice_1"]]
        
        temp_nif_path = "temp_test_image.nif"
        
        try:
            # Salva o arquivo binário
            save_nif_file(temp_nif_path, width, height, quality_int, z_shape, z_string, y_strings)
            
            # Carrega o arquivo binário
            w_rec, h_rec, q_rec, z_shape_rec, z_string_rec, y_strings_rec = load_nif_file(temp_nif_path)
            
            # Compara
            self.assertEqual(w_rec, width)
            self.assertEqual(h_rec, height)
            self.assertEqual(q_rec, quality_int)
            self.assertEqual(z_shape_rec, z_shape)
            self.assertEqual(z_string_rec, z_string)
            self.assertEqual(y_strings_rec, y_strings)
            
        finally:
            if os.path.exists(temp_nif_path):
                os.remove(temp_nif_path)

if __name__ == '__main__':
    unittest.main()
