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
##  Resultados Obtidos (Avaliado no Benchmark Kodak24)
O modelo foi treinado por 300 épocas no dataset de alta resolução **DIV2K** e avaliado no benchmark padrão da literatura **Kodak24** (completamente disjunto do conjunto de treino), apresentando os seguintes resultados médios agregados ($\text{média} \pm \text{desvio padrão}$):

| Qualidade $q$ | Bitrate (bpp) | PSNR (dB) | MS-SSIM | LPIPS |
| :--- | :--- | :--- | :--- | :--- |
| **0.10** | $0.8311 \pm 0.1220$ | $28.77 \pm 2.00$ | $0.98363 \pm 0.00389$ | $0.05109 \pm 0.01481$ |
| **0.30** | $0.8393 \pm 0.1224$ | $28.77 \pm 2.00$ | $0.98373 \pm 0.00388$ | $0.05086 \pm 0.01489$ |
| **0.50** | $0.8506 \pm 0.1228$ | $28.77 \pm 2.00$ | $0.98382 \pm 0.00388$ | $0.05065 \pm 0.01477$ |
| **0.70** | $0.8644 \pm 0.1231$ | $28.77 \pm 2.00$ | $0.98393 \pm 0.00390$ | $0.05048 \pm 0.01472$ |
| **0.90** | $0.8831 \pm 0.1240$ | $28.77 \pm 2.00$ | $0.98407 \pm 0.00391$ | $0.05043 \pm 0.01488$ |

* **Reprodutibilidade Científica**: A tabela acima é obtida diretamente executando a avaliação integrada:
  ```bash
  python scripts/eval.py --checkpoint checkpoints/nif_epoch_300.pth --image kodak24/ --aggregate --save_output
  ```
---
##  Dashboard Web Interativo
O projeto acompanha um dashboard web local construído em Flask para demonstrações visuais e análises em tempo real:
* **Slide-to-Compare**: Comparador deslizante vertical interativo para visualizar lado a lado a imagem Original e a Reconstruída pela IA.
* **Estatísticas em Tempo Real**: Exibição instantânea do ganho de compressão, taxa (bpp), PSNR e MS-SSIM.
* **Download Binário**: Compactação e descompactação completas via drag-and-drop.
---
##  Como Executar

### 1. Instalação das Dependências

Garante a instalação do PyTorch, CompressAI e dependências do dashboard:
```bash
pip install -r requirements.txt
pip install Flask
```
2. Treinamento do Modelo
Para treinar o modelo do zero usando o dataset DIV2K:

```bash
python scripts/train.py --dataset DIV2K_train_HR --epochs 300
```
3. Compactação e Descompactação via CLI

Para compactar uma imagem:
```bash
python nif_tool.py compress --checkpoint checkpoints/nif_epoch_300.pth --input NOME_DA_IMAGEM.png --output imagem_comprimida.nif --quality 0.5
```
Para descompactar o arquivo .nif:
```bash
python nif_tool.py decompress --checkpoint checkpoints/nif_epoch_300.pth --input imagem_comprimida.nif --output reconstruida.png
```
4. Abrir o Dashboard Interativo

Para iniciar a interface web:
```bash
python scripts/dashboard.py
```
Em seguida, abra o navegador em: http://localhost:5000

---

##  Latência & Custo Operacional (AWS g4dn.xlarge)

O script de benchmark mede a velocidade de compressão/descompressão do codec NIF e estima o custo para processar **1 Milhão de imagens** em uma instância de referência na nuvem (AWS EC2 `g4dn.xlarge`, custo de **$0.526 por hora**):

| Dispositivo | Resolução | Encode (ms) | Decode (ms) | Total (ms) | Custo / 1M Imagens (USD) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CPU** | 256x256 | 375.18 | 255.60 | 630.79 | **$92.16** |
| **CPU** | 512x512 | 1212.72 | 880.88 | 2093.59 | **$305.90** |
| **CPU** | 1024x1024 | 6461.80 | 5582.96 | 12044.76 | **$1759.87** |
| **GPU (CUDA)** | 256x256 | 406.20 | 245.09 | 651.29 | **$95.16** |
| **GPU (CUDA)** | 512x512 | 1141.65 | 604.92 | 1746.57 | **$255.19** |
| **GPU (CUDA)** | 1024x1024 | 4427.06 | 2222.63 | 6649.69 | **$971.59** |

Para rodar este benchmark em sua máquina:
```bash
python scripts/benchmark_latency.py
```

---

##  Limitações & Próximos Passos
O formato NIF é um projeto experimental de pesquisa em compressão neural de imagens e possui as seguintes limitações conhecidas:
1. **Curva RD Fina**: Atualmente, a curva de taxa-distorção apresenta modulação estreita (*Conditioning Collapse*) na variação de qualidade $q$. Para apresentações profissionais, recomenda-se a aplicação de *Positional Encoding* no parâmetro de qualidade $q$ e taxas de aprendizado maiores na MLP de condicionamento na próxima sessão de treino.
2. **Dependência de Checkpoints**: Assim como todo formato neural, a descompressão requer exatamente o mesmo modelo e pesos de treino que comprimiram a imagem.
3. **Espaço de Cor**: Atualmente operamos em espaço de cor RGB puro de 24-bits. Uma implementação futura integrará YCbCr 4:2:0 para melhorar em mais 15-20% as taxas de bitrate.
