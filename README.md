# NIF — Neural Image Format 🚀

[![Python Package](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/python-package.yml/badge.svg)](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/python-package.yml)
[![Pylint](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/pylint.yml/badge.svg)](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/pylint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

O **NIF (Neural Image Format)** é um formato e codec de compressão de imagens de ponta a ponta baseado em redes neurais profundas (*Learned Image Compression*). Desenvolvido para superar as limitações de artefatos em blocos dos codecs legados, o NIF combina modulação contínua de taxa (*FiLM Conditioning*), modelagem de entropia híbrida (Channel-wise + Spatial Checkerboard), **DPCM Noise Shaping no espaço latente** e decodificação altamente paralela.

---

## 🌟 Principais Inovações Tecnológicas

1. **Modulação Contínua de Taxa de Bits (FiLM-Conditioned)**:
   * Controle contínuo de qualidade ($q \in [0.1, 1.0]$) com um único conjunto de pesos treinado, eliminando a necessidade de múltiplos modelos para diferentes qualidades.
   * Restrição *Sigmoid Latent Scaling* garantindo estabilidade de variância zero (**Zero NaNs** em imagens uniformes/planas).
2. **Noise Shaping DPCM no Espaço Latente**:
   * Algoritmo de realocação de resíduo de quantização entre fatias latentes (*Slices*), gerando **+0.42 dB de PSNR** e **-2.86% de redução real de tamanho** (*free lunch* de taxa-distorção).
3. **Decodificação Paralela em Checkerboard (202 ms)**:
   * Modelo de entropia híbrido com predição em xadrez espacial que reduz os passos de contexto autoregressivo para apenas 2 iterações paralelas por fatia (**aceleração de 10.96x** sobre o autoregressivo clássico).
4. **Perda Perceptual Mascarada por Fidelidade Estrutural (SFM)**:
   * Combinação dinâmica de MSE, MS-SSIM e LPIPS guiada por mapa de bordas para sintetizar texturas naturais sem distorcer caracteres ou estruturas críticas.

---

## 📊 Resultados e Benchmark Rigoroso (Dataset Kodak24)

Todos os resultados foram medidos de forma pareada nas 24 imagens do dataset **Kodak24**, com avaliação **interpolada no mesmo bitrate (*Bitrate-Matched*)** contra codecs industriais consolidados:

### Comparação no Ponto Central ($0.80\text{ bpp}$ no Kodak24)

| Codec / Variante | Bitrate | PSNR (dB) | MS-SSIM | LPIPS (menor = melhor) | Latência Decode | Posicionamento Técnico |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **NIF v4 (`Perceptual`)** | **0.80 bpp** | 28.15 dB | 0.9796 | **`0.0828`** | **`~202 ms`** | **+26.4% superior ao JPEG em LPIPS** |
| **NIF Config A (`Balanced`)** | **0.80 bpp** | 28.40 dB | **`0.9804`** | **`0.0953`** | **`~202 ms`** | **Fronteira de Pareto Ótima** (+15.3% LPIPS & +0.0094 SSIM vs JPEG) |
| **NIF Config D (`High-PSNR`)** | **0.80 bpp** | **29.76 dB** | 0.9781 | 0.1453 | **`~202 ms`** | Máximo PSNR da CNN atual (+1.61 dB vs v4) |
| *JPEG (Padrão)* | *0.80 bpp* | *31.45 dB* | *0.9710* | *0.1125* | *~15 ms* | Linha de Base Clássica DCT |
| *WebP (Google)* | *0.80 bpp* | *33.77 dB* | *0.9774* | *0.0901* | *~35 ms* | Linha de Base WebP |
| *AVIF (AOMedia AV1)* | *0.80 bpp* | *34.56 dB* | *0.9838* | *0.0736* | *~450 ms* | Estado da Arte da Indústria |

> **Diagnóstico**: O NIF entrega **qualidade visual perceptual e integridade estrutural substancialmente superiores ao JPEG** (+15% a +26% de redução de distorção LPIPS, sem blocos 8x8 visíveis) com decodificação rápida (~202 ms). Codecs ultra-pesados como o AVIF alcançam maior PSNR em decorrência de intra-predição espacial de alta complexidade.

---

## ⚡ Latência & Custo Operacional em Nuvem (AWS EC2 g4dn.xlarge)

Medição empírica de latência e projeção de custo para processamento em larga escala (**1 Milhão de Imagens**) na instância de referência AWS EC2 `g4dn.xlarge` ($0.526/hora):

| Resolução | Hardware | Tempo Encode (ms) | Tempo Decode (ms) | Tempo Total (ms) | Custo / 1 Milhão Imagens (USD) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **256x256** | CPU | 375.2 ms | 255.6 ms | 630.8 ms | **$92.16** |
| **512x512** | CPU | 1212.7 ms | 880.9 ms | 2093.6 ms | **$305.90** |
| **1024x1024** | CPU | 6461.8 ms | 5582.9 ms | 12044.8 ms | **$1759.87** |
| **256x256** | **GPU (CUDA)** | 104.1 ms | **202.8 ms** | **306.8 ms** | **$44.91** |
| **512x512** | **GPU (CUDA)** | 114.8 ms | **208.7 ms** | **323.5 ms** | **$47.36** |
| **1024x1024** | **GPU (CUDA)** | 104.1 ms | **202.8 ms** | **306.8 ms** | **$44.91** *(11x mais barato que CPU!)* |

---

## 🛠️ Instalação e Uso Rápido

### 1. Clonar e Instalar Dependências
```bash
git clone https://github.com/gabrielphilipus/NIF-Neural-Image-Format.git
cd NIF-Neural-Image-Format
pip install -r requirements.txt
```

### 2. Compactar Imagem (.png/.jpg $\to$ .nif)
```bash
python nif_tool.py compress --checkpoint checkpoints_v4_production/nif_epoch_300.pth --input foto.png --output foto.nif --quality 0.5
```

### 3. Descompactar Imagem (.nif $\to$ .png)
```bash
python nif_tool.py decompress --checkpoint checkpoints_v4_production/nif_epoch_300.pth --input foto.nif --output foto_reconstruida.png
```

### 4. Dashboard Web Interativo (Slide-to-Compare)
Para abrir o comparador visual lado a lado no navegador:
```bash
python scripts/dashboard.py
```
Acesse em seu navegador: `http://localhost:5000`

---

## 📦 Model Zoo & Checkpoints Pré-Treinados

Os pesos treinados dos modelos estão disponíveis para download na aba de [Releases](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/releases):

| Modelo | Arquivo | Descrição | Casos de Uso Recomendados |
|:---|:---|:---|:---|
| **`NIF-Perceptual (v4)`** | `nif_v4_perceptual_300ep.pth` | Treinado por 300 épocas com LPIPS e Discriminador PatchGAN | Redes sociais, web, fotografia móvel e thumbnails |
| **`NIF-Balanced (Config A)`** | `nif_config_A_balanced.pth` | Otimizado na fronteira de Pareto (40% MSE, 40% SSIM, 20% LPIPS) | Uso geral, balanceamento entre nitidez e PSNR |
| **`NIF-Fidelity (Config D)`** | `nif_config_D_fidelity.pth` | Treinado por 50 épocas sob MSE puro | Imagens técnicas, satelitais e médicas |

---

## 🧪 Testes Unitários e Integração Contínua (CI)

O repositório possui uma suite de testes unitários automatizados cobrindo a integridade do compressor, estabilidade de variância zero e serialização de cabeçalhos binários:
```bash
python -m unittest discover tests
```

---

## 📄 Licença

Este projeto é disponibilizado sob a licença [MIT](LICENSE).
