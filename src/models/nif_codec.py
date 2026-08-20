import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import EntropyBottleneck
from compressai.layers import GDN
from .entropy_model import ChannelCheckerboardEntropyModel

class QualityConditioningNetwork(nn.Module):
    """
    Rede MLP que converte o parâmetro de qualidade 'q' (0.1 a 1.0)
    em fatores de escala (scale) e translação (bias) para as camadas FiLM.
    """
    def __init__(self, out_features):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_features * 2)
        )
        
    def forward(self, q):
        # q: [B, 1]
        out = self.fc(q)
        scale, bias = out.chunk(2, dim=1)
        # Reshape para broadcast em [B, C, H, W]
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        bias = bias.unsqueeze(-1).unsqueeze(-1)
        return scale, bias


class FiLMBlock(nn.Module):
    """
    Aplica modulação linear de características (Feature-wise Linear Modulation)
    nos mapas de características.
    """
    def forward(self, x, scale, bias):
        return x * (1.0 + scale) + bias


class NIFCodec(CompressionModel):
    """
    Neural Image Format (NIF) Codec.
    - Codificador/Decodificador com modulação de taxa (FiLM) condicionada em 'q'.
    - Modelo de entropia híbrido (Channel-wise + Spatial Checkerboard).
    - Gerador de Máscara de Fidelidade Estrutural (SFM) para queda generativa seletiva.
    """
    def __init__(self, num_filters=128, latent_dim=192, num_slices=8):
        super().__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.num_slices = num_slices

        # 1. Redes de condicionamento de qualidade
        self.cond_enc1 = QualityConditioningNetwork(num_filters)
        self.cond_enc2 = QualityConditioningNetwork(num_filters)
        self.cond_enc3 = QualityConditioningNetwork(num_filters)
        self.cond_enc4 = QualityConditioningNetwork(latent_dim)

        self.cond_dec1 = QualityConditioningNetwork(num_filters)
        self.cond_dec2 = QualityConditioningNetwork(num_filters)
        self.cond_dec3 = QualityConditioningNetwork(num_filters)
        self.cond_dec4 = QualityConditioningNetwork(3)

        # 2. Análise (Encoder)
        self.enc_conv1 = nn.Conv2d(3, num_filters, kernel_size=5, stride=2, padding=2)
        self.enc_gdn1 = GDN(num_filters)
        self.enc_film1 = FiLMBlock()

        self.enc_conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)
        self.enc_gdn2 = GDN(num_filters)
        self.enc_film2 = FiLMBlock()

        self.enc_conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)
        self.enc_gdn3 = GDN(num_filters)
        self.enc_film3 = FiLMBlock()

        self.enc_conv4 = nn.Conv2d(num_filters, latent_dim, kernel_size=5, stride=2, padding=2)
        self.enc_film4 = FiLMBlock()

        # 3. Hiperprior (Análise do Latente)
        self.hyper_enc_conv1 = nn.Conv2d(latent_dim, num_filters, kernel_size=3, stride=1, padding=1)
        self.hyper_enc_relu1 = nn.ReLU(inplace=True)
        self.hyper_enc_conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)
        self.hyper_enc_relu2 = nn.ReLU(inplace=True)
        self.hyper_enc_conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2)
        
        # Bottleneck de entropia para codificar a hyperprior 'z'
        self.entropy_bottleneck = EntropyBottleneck(num_filters)

        # 4. Hiperprior (Síntese do Latente)
        self.hyper_dec_deconv1 = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.hyper_dec_relu1 = nn.ReLU(inplace=True)
        self.hyper_dec_deconv2 = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.hyper_dec_relu2 = nn.ReLU(inplace=True)
        self.hyper_dec_conv3 = nn.Conv2d(num_filters, latent_dim, kernel_size=3, stride=1, padding=1)

        # 5. Modelo de Entropia Avançado (Channel + Spatial Checkerboard)
        self.entropy_model = ChannelCheckerboardEntropyModel(
            in_channels=latent_dim, num_slices=num_slices, latent_dim=latent_dim
        )

        # 6. Síntese (Decoder)
        self.dec_deconv1 = nn.ConvTranspose2d(latent_dim, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_igdn1 = GDN(num_filters, inverse=True)
        self.dec_film1 = FiLMBlock()

        self.dec_deconv2 = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_igdn2 = GDN(num_filters, inverse=True)
        self.dec_film2 = FiLMBlock()

        self.dec_deconv3 = nn.ConvTranspose2d(num_filters, num_filters, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_igdn3 = GDN(num_filters, inverse=True)
        self.dec_film3 = FiLMBlock()

        self.dec_deconv4 = nn.ConvTranspose2d(num_filters, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_film4 = FiLMBlock()

    def compute_structural_mask(self, x):
        """
        Gera um mapa de densidade de alta frequência (bordas) para identificar
        textos e detalhes estruturais. Normalizado no intervalo [0, 1].
        """
        # Converte para tons de cinza
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        
        # Filtro laplaciano fixo
        laplacian_kernel = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32, device=x.device).view(1, 1, 3, 3)
        
        edges = torch.abs(nn.functional.conv2d(gray, laplacian_kernel, padding=1))
        
        # Suaviza localmente
        edges_smoothed = nn.functional.avg_pool2d(edges, kernel_size=5, stride=1, padding=2)
        
        # Normalização estável por imagem
        max_val = torch.amax(edges_smoothed, dim=(2, 3), keepdim=True) + 1e-5
        sfm = edges_smoothed / max_val
        return sfm

    def forward(self, x, quality):
        """
        Grafo de computação do treinamento.
        quality: Tensor de forma [B, 1] com valores em [0.1, 1.0]
        """
        # A. Gerar máscara de fidelidade estrutural
        sfm = self.compute_structural_mask(x)

        # B. Encoder (Análise) com modulação de qualidade
        scale_e1, bias_e1 = self.cond_enc1(quality)
        x_e1 = self.enc_film1(self.enc_gdn1(self.enc_conv1(x)), scale_e1, bias_e1)

        scale_e2, bias_e2 = self.cond_enc2(quality)
        x_e2 = self.enc_film2(self.enc_gdn2(self.enc_conv2(x_e1)), scale_e2, bias_e2)

        scale_e3, bias_e3 = self.cond_enc3(quality)
        x_e3 = self.enc_film3(self.enc_gdn3(self.enc_conv3(x_e2)), scale_e3, bias_e3)

        scale_e4, bias_e4 = self.cond_enc4(quality)
        y = self.enc_film4(self.enc_conv4(x_e3), scale_e4, bias_e4)

        # C. Hyperprior (Análise do Latente)
        z = self.hyper_enc_conv3(
            self.hyper_enc_relu2(
                self.hyper_enc_conv2(
                    self.hyper_enc_relu1(
                        self.hyper_enc_conv1(y)
                    )
                )
            )
        )

        # D. Quantização e cálculo de probabilidade da Hyperprior
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        # E. Decoder da Hyperprior (Síntese)
        hyper_features = self.hyper_dec_conv3(
            self.hyper_dec_relu2(
                self.hyper_dec_deconv2(
                    self.hyper_dec_relu1(
                        self.hyper_dec_deconv1(z_hat)
                    )
                )
            )
        )

        # F. Modelo de Entropia (Channel + Checkerboard) e quantização do latente principal
        y_hat, y_likelihoods = self.entropy_model(y, hyper_features)

        # G. Decoder (Síntese) com modulação de qualidade
        scale_d1, bias_d1 = self.cond_dec1(quality)
        y_d1 = self.dec_film1(self.dec_igdn1(self.dec_deconv1(y_hat)), scale_d1, bias_d1)

        scale_d2, bias_d2 = self.cond_dec2(quality)
        y_d2 = self.dec_film2(self.dec_igdn2(self.dec_deconv2(y_d1)), scale_d2, bias_d2)

        scale_d3, bias_d3 = self.cond_dec3(quality)
        y_d3 = self.dec_film3(self.dec_igdn3(self.dec_deconv3(y_d2)), scale_d3, bias_d3)

        scale_d4, bias_d4 = self.cond_dec4(quality)
        x_hat = self.dec_film4(self.dec_deconv4(y_d3), scale_d4, bias_d4)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
                "z": z_likelihoods
            },
            "sfm": sfm
        }

    def load_state_dict(self, state_dict, strict=True):
        # Sobrescreve para compatibilidade com CompressAI
        super().load_state_dict(state_dict, strict=strict)

    def compress(self, x, quality):
        """
        Codifica aritmeticamente a imagem 'x' para uma representação binária estruturada.
        quality: tensor [1, 1] contendo o valor real de q em [0.1, 1.0].
        """
        # A. Encoder (Análise)
        scale_e1, bias_e1 = self.cond_enc1(quality)
        x_e1 = self.enc_film1(self.enc_gdn1(self.enc_conv1(x)), scale_e1, bias_e1)

        scale_e2, bias_e2 = self.cond_enc2(quality)
        x_e2 = self.enc_film2(self.enc_gdn2(self.enc_conv2(x_e1)), scale_e2, bias_e2)

        scale_e3, bias_e3 = self.cond_enc3(quality)
        x_e3 = self.enc_film3(self.enc_gdn3(self.enc_conv3(x_e2)), scale_e3, bias_e3)

        scale_e4, bias_e4 = self.cond_enc4(quality)
        y = self.enc_film4(self.enc_conv4(x_e3), scale_e4, bias_e4)

        # B. Hyperprior (Análise do Latente)
        z = self.hyper_enc_conv3(
            self.hyper_enc_relu2(
                self.hyper_enc_conv2(
                    self.hyper_enc_relu1(
                        self.hyper_enc_conv1(y)
                    )
                )
            )
        )

        # C. Comprimir Hyperprior z
        z_strings = self.entropy_bottleneck.compress(z)
        
        # D. Descomprimir z localmente para gerar as previsões do modelo de entropia
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        hyper_features = self.hyper_dec_conv3(
            self.hyper_dec_relu2(
                self.hyper_dec_deconv2(
                    self.hyper_dec_relu1(
                        self.hyper_dec_deconv1(z_hat)
                    )
                )
            )
        )

        # E. Comprimir latente principal y usando o contexto
        y_strings = self.entropy_model.compress(y, hyper_features)

        return {
            "strings": [z_strings, y_strings],
            "shape": z.shape[-2:]  # Necessário para restaurar z
        }

    def decompress(self, strings, shape, quality):
        """
        Decodifica a representação binária estruturada de volta para a imagem reconstruída.
        """
        z_strings, y_strings = strings
        
        # A. Decodificar Hyperprior z_hat
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        
        # B. Obter previsões do modelo de entropia a partir de z_hat
        hyper_features = self.hyper_dec_conv3(
            self.hyper_dec_relu2(
                self.hyper_dec_deconv2(
                    self.hyper_dec_relu1(
                        self.hyper_dec_deconv1(z_hat)
                    )
                )
            )
        )

        # C. Decodificar y_hat usando o contexto condicionado
        # A resolução de y é 4 vezes a de z (nosso encoder tem 4 subamostragens e a hyperprior tem 2 adicionais)
        H_y, W_y = shape[0] * 4, shape[1] * 4
        y_hat = self.entropy_model.decompress(y_strings, hyper_features, H_y, W_y)

        # D. Decoder (Síntese)
        scale_d1, bias_d1 = self.cond_dec1(quality)
        y_d1 = self.dec_film1(self.dec_igdn1(self.dec_deconv1(y_hat)), scale_d1, bias_d1)

        scale_d2, bias_d2 = self.cond_dec2(quality)
        y_d2 = self.dec_film2(self.dec_igdn2(self.dec_deconv2(y_d1)), scale_d2, bias_d2)

        scale_d3, bias_d3 = self.cond_dec3(quality)
        y_d3 = self.dec_film3(self.dec_igdn3(self.dec_deconv3(y_d2)), scale_d3, bias_d3)

        scale_d4, bias_d4 = self.cond_dec4(quality)
        x_hat = self.dec_film4(self.dec_deconv4(y_d3), scale_d4, bias_d4)

        return {
            "x_hat": x_hat
        }
