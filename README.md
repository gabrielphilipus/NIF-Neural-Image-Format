# NIF - Neural Image Format 

[![Python Package](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/python-package.yml/badge.svg)](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/python-package.yml)
[![Pylint](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/pylint.yml/badge.svg)](https://github.com/gabrielphilipus/NIF-Neural-Image-Format/actions/workflows/pylint.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

O **NIF (Neural Image Format)** é um codec de compressão de imagem baseado em **aprendizado profundo de ponta a ponta (End-to-End Learned Image Compression)**. Ele utiliza redes neurais convolucionais profundas acopladas a mecanismos de atenção, modulação de taxa variável e codificação de entropia avançada para superar as limitações dos formatos tradicionais.

---

##  Destaques & Arquitetura

O codec foi estruturado com as tecnologias mais recentes em compressão neural de imagens:

1. **Codificador/Decodificador Variencial (FiLM-Conditioned)**: Suporta compressão em taxa variável com um único modelo treinado, controlando o nível de qualidade ($q$) continuamente via camadas de modulação FiLM.
2. **Mecanismo de Atenção por Canal (Squeeze-and-Excitation)**: Integrado em todas as 4 escalas do Encoder e Decoder para guiar dinamicamente a alocação de bits para regiões com texturas mais complexas (bordas, textos, etc.).
3. **Filtro de Loop Neural Residual (NLF)**: Um pós-processador leve baseado em blocos residuais acoplado ao final do decodificador para refinar a imagem de saída e eliminar artefatos residuais de reconstrução.
4. **Modelo de Entropia Checkerboard + GMM**:
   * Abordagem autoregressiva em xadrez espacial (Checkerboard) combinada com contexto entre canais (Channel-wise).
   * Substituição do modelo Gaussiano simples por **GMM (Gaussian Mixture Model)** com $K=3$ componentes para modelar distribuições hiper-complexas.
   * Codificador/Decodificador aritmético (ANS) embarcado nativamente em C++.

---

##  Resultados Obtidos (Benchmark Kodak24 Completo - 24 Imagens)

O modelo foi avaliado no benchmark padrão da literatura **Kodak24** utilizando a flag `--aggregate` do script de avaliação. Apresentamos abaixo os resultados consolidados do **Novo Modelo (Checkpoint 250)** em comparação com o **Modelo Original (Checkpoint 265 antigo)** que sofria de colapso de condicionamento (*Conditioning Collapse*):

### Tabela de Comparação de Desempenho (Média ± Desvio Padrão)

| Qualidade $q$ | Checkpoint Original (Antes - Colapso) | Checkpoint E-300 (Depois - Otimizado) |
| :---: | :--- | :--- |
| **0.10** | **Bpp:** $0.8438 \pm 0.129$ <br>**PSNR:** $27.97 \pm 2.48$ dB <br>**MS-SSIM:** $0.9790 \pm 0.006$ <br>**LPIPS:** $0.0584 \pm 0.020$ | **Bpp:** $0.7130 \pm 0.052$ <br>**PSNR:** $27.91 \pm 2.45$ dB <br>**MS-SSIM:** $0.9735 \pm 0.006$ <br>**LPIPS:** $0.0673 \pm 0.022$ |
| **0.30** | **Bpp:** $0.8526 \pm 0.130$ <br>**PSNR:** $27.98 \pm 2.48$ dB <br>**MS-SSIM:** $0.9791 \pm 0.006$ <br>**LPIPS:** $0.0583 \pm 0.020$ | **Bpp:** $0.8371 \pm 0.065$ <br>**PSNR:** $28.20 \pm 2.48$ dB <br>**MS-SSIM:** $0.9798 \pm 0.005$ <br>**LPIPS:** $0.0576 \pm 0.021$ |
| **0.50** | **Bpp:** $0.8622 \pm 0.130$ <br>**PSNR:** $27.97 \pm 2.48$ dB <br>**MS-SSIM:** $0.9792 \pm 0.006$ <br>**LPIPS:** $0.0582 \pm 0.020$ | **Bpp:** $0.9831 \pm 0.083$ <br>**PSNR:** $28.34 \pm 2.50$ dB <br>**MS-SSIM:** $0.9830 \pm 0.005$ <br>**LPIPS:** $0.0527 \pm 0.020$ |
| **0.70** | **Bpp:** $0.8735 \pm 0.130$ <br>**PSNR:** $27.97 \pm 2.48$ dB <br>**MS-SSIM:** $0.9794 \pm 0.006$ <br>**LPIPS:** $0.0580 \pm 0.020$ | **Bpp:** $1.1673 \pm 0.101$ <br>**PSNR:** $28.41 \pm 2.51$ dB <br>**MS-SSIM:** $0.9846 \pm 0.004$ <br>**LPIPS:** $0.0508 \pm 0.019$ |
| **0.90** | **Bpp:** $0.8879 \pm 0.131$ <br>**PSNR:** $27.97 \pm 2.47$ dB <br>**MS-SSIM:** $0.9794 \pm 0.006$ <br>**LPIPS:** $0.0580 \pm 0.020$ | **Bpp:** $1.3669 \pm 0.123$ <br>**PSNR:** $28.45 \pm 2.51$ dB <br>**MS-SSIM:** $0.9854 \pm 0.004$ <br>**LPIPS:** $0.0501 \pm 0.019$ |

### Principais Conclusões dos Resultados:
* **Mitigação do Conditioning Collapse:** A variação do bitrate médio entre $q=0.10$ e $q=0.90$ aumentou de **5.2%** (no modelo antigo colapsado) para **91.7%** (no novo modelo de produção), representando um aumento de **~17.6x na sensibilidade** da curva de taxa de bits.
* **Comportamento Monótono de Qualidade:** A fidelidade (PSNR e MS-SSIM) agora cresce de forma dinâmica e monótona em função de $q$.
* **Alocação Espacial Eficiente**: O modelo agora redistribui bits espacialmente (Importance Map) e compensa ruído (Noise Shaping DPCM), resultando em melhoria visível de detalhes.

---

##  Como Executar as Avaliações e Benchmarks

### 1. Script de Avaliação (`eval.py`)
Para rodar a avaliação do modelo de forma agregada sobre todas as imagens do dataset Kodak24:
```bash
python scripts/eval.py --checkpoint checkpoints_v3_production/nif_epoch_300.pth --image kodak24/ --aggregate --save_output
```

### 2. Script de Benchmark Comparativo (`benchmark.py`)
O script de benchmark foi estendido para rodar de forma automática tanto em arquivos individuais quanto em diretórios inteiros (em lote). 

Para avaliar e gerar curvas Rate-Distortion comparativas com **JPEG e WebP** com barras de erro do desvio padrão ($\pm$ std):
```bash
python scripts/benchmark.py --checkpoint checkpoints_v3_production/nif_epoch_300.pth --image kodak24 --output_dir results/benchmark_epoch_300
```
Isso gerará os arquivos:
* `results/benchmark_epoch_300/benchmark_aggregated.json` (pontos brutos e estatísticas agregadas).
* `rd_curve_psnr.png`, `rd_curve_msssim.png` e `rd_curve_lpips.png` contendo as curvas de Rate-Distortion.

---

## ↳ Dashboard Web Interativo

O projeto acompanha um dashboard web local construído em Flask para demonstrações visuais e análises em tempo real:
* **Slide-to-Compare**: Comparador deslizante vertical interativo para visualizar lado a lado a imagem Original e a Reconstruída pela IA.
* **Estatísticas em Tempo Real**: Exibição instantânea do ganho de compressão, taxa (bpp), PSNR e MS-SSIM.
* **Download Binário**: Compactação e descompactação completas via drag-and-drop.

---

## ↳ Como Executar o Projeto Localmente

### 1. Instalação das Dependências
Garante a instalação do PyTorch, CompressAI e dependências do dashboard:
```bash
pip install -r requirements.txt
pip install Flask
```

### 2. Treinamento do Modelo
Para treinar o modelo do zero usando o dataset DIV2K:
```bash
python scripts/train.py --dataset DIV2K_train_HR --epochs 300
```

### 3. Compactação e Descompactação via CLI
Para compactar uma imagem:
```bash
python nif_tool.py compress --checkpoint checkpoints_v3_production/nif_epoch_300.pth --input NOME_DA_IMAGEM.png --output imagem_comprimida.nif --quality 0.5
```
Para descompactar o arquivo `.nif`:
```bash
python nif_tool.py decompress --checkpoint checkpoints_v3_production/nif_epoch_300.pth --input imagem_comprimida.nif --output reconstruida.png
```

### 4. Abrir o Dashboard Interativo
Para iniciar a interface web:
```bash
python scripts/dashboard.py
```
Em seguida, abra o navegador em: `http://localhost:5000`

---

##  Latência & Custo Operacional (AWS g4dn.xlarge)

O script de benchmark mede a velocidade de compressão/descompressão do codec NIF e estima o custo para processar **1 Milhão de imagens** em uma instância de referência na nuvem (AWS EC2 `g4dn.xlarge`, custo de **$0.526 por hora**):

| Dispositivo | Resolução | Encode (ms) | Decode (ms) | Total (ms) | Custo / 1M Imagens (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU** | 256x256 | 375.18 | 255.60 | 630.79 | **$92.16** |
| **CPU** | 512x512 | 1212.72 | 880.88 | 2093.59 | **$305.90** |
| **CPU** | 1024x1024 | 6461.80 | 5582.96 | 12044.76 | **$1759.87** |
| **GPU (CUDA)** | 256x256 | 104.06 | 202.76 | 306.82 | **$44.91** |
| **GPU (CUDA)** | 512x512 | 114.84 | 208.68 | 323.52 | **$47.36** |
| **GPU (CUDA)** | 1024x1024 | 104.06 | **202.76** | 306.82 | **$44.91** (11x mais barato!) |

Para rodar este benchmark em sua máquina:
```bash
python scripts/benchmark_latency.py
```

---

## ↳ Comparação Honesta com o Estado da Arte (JPEG Baseline)

Para calibrar as expectativas de progresso do NIF em relação a formatos maduros do mercado:
* No bitrate mais alto testado (**$q=0.90$ com $\approx 1.37\text{ bpp}$**), o NIF entrega **$28.45\text{ dB}$ de PSNR** e **$0.0501$ de LPIPS**.
* No mesmo bitrate exato de **$1.368\text{ bpp}$**, o **JPEG (em $q=75$)** atinge **$34.52\text{ dB}$ de PSNR** e **$0.0240$ de LPIPS**.

> [!IMPORTANT]
> O JPEG (um codec clássico de 1992) ainda supera o NIF por **$6.07\text{ dB}$ em PSNR** e **$2\times$ em LPIPS**. Isso ocorre devido às limitações de tamanho de parâmetros da nossa rede móvel e do pipeline inicial. O objetivo principal do NIF é validar o controle dinâmico de taxa e latência em redes neurais de ponta-a-ponta, e não paridade imediata de produção com codecs legados consolidados em hardware.

---

## ↳ Limitações & Possíveis Melhorias
O formato NIF é um projeto experimental de pesquisa em compressão neural de imagens e possui as seguintes limitações conhecidas:
1. **Ineficiência FiLM em Extremos**: As MLPs convolucionais FiLM (`cond_enc`/`cond_dec`) demonstram aprendizado fraco no desvio padrão de pesos ($\approx 0.08$), limitando o ganho de PSNR absoluto em bpp alto. Recomenda-se aplicar taxas de aprendizado ainda maiores especificamente na MLP FiLM em futuros treinos.
2. **Dependência de Checkpoints**: Assim como todo formato neural, a descompressão requer exatamente o mesmo modelo e pesos de treino que comprimiram a imagem.
3. **Espaço de Cor**: Atualmente operamos em espaço de cor RGB puro de 24-bits. Uma implementação futura integrará YCbCr 4:2:0 para melhorar em mais 15-20% as taxas de bitrate.
