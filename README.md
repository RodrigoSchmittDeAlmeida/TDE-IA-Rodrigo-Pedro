# T1 – Tic Tac Toe com ML

Disciplina de Inteligência Artificial – PUCRS (Profa. Silvia Moraes)
Autores: Rodrigo Schmitt de Almeida e Pedro Filipetto

Sistema de IA que recebe o estado de um tabuleiro 3x3 do jogo da velha e o classifica em uma de quatro classes:

- **Tem jogo**
- **Jogador X venceu**
- **Jogador O venceu**
- **Empate**

A IA não joga: ela apenas verifica o estado do jogo.

> **Situação do trabalho:** dataset, pré-processamento, divisão dos dados e os cinco algoritmos estão implementados e avaliados (seções 1 a 6). O front end, os gráficos, o vídeo e a conclusão final ainda estão pendentes (seção 8).

## Como executar

```bash
cd TIC-TAC-TOE
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python jogo.py
```

A execução leva cerca de 30 segundos. O script precisa ser executado de dentro da pasta `TIC-TAC-TOE`, pois lê `tic-tac-toe.data.csv` por caminho relativo. A simulação usa `random.seed(42)` e os divisores usam `random_state=42`. Por isso os resultados são reproduzíveis: só os tempos de treino variam entre execuções.

## Estrutura do repositório

| Arquivo | Conteúdo |
|---|---|
| `TIC-TAC-TOE/jogo.py` | Dataset, pré-processamento, divisão dos dados e todos os algoritmos |
| `TIC-TAC-TOE/tic-tac-toe.data.csv` | Dataset original da UCI (958 tabuleiros finais) |
| `TIC-TAC-TOE/tic-tac-toe.names.csv`, `Index.csv` | Documentação original da UCI |
| `TIC-TAC-TOE/requirements.txt` | Dependências (numpy, pandas, scikit-learn) |
| `resultados.txt` | Saída de uma execução anterior (somente k-NN, sem seed fixa) |

---

## 1. Introdução e objetivo

O trabalho constrói um classificador de estado de jogo. A entrada é um tabuleiro 3x3 com casas `X`, `O` ou vazias, e a saída é uma das quatro classes acima. Foram comparados cinco algoritmos e duas formas de representar o tabuleiro. O objetivo é descobrir qual combinação é mais adequada e menos custosa para o problema.

## 2. Dataset

### 2.1 Fonte

O dataset é o *Tic-Tac-Toe Endgame* da UCI (https://archive.ics.uci.edu/dataset/101/tic+tac+toe+endgame). Ele tem **958 instâncias**, cada uma com 9 atributos (as casas `TL, TM, TR, ML, MM, MR, BL, BM, BR`, com valores `x`, `o` ou `b`) e uma classe `positive/negative`.

### 2.2 Problemas encontrados

| # | Problema | Consequência |
|---|---|---|
| 1 | A classe original é binária (`positive` = X venceu; `negative` = não venceu). O problema pede 4 classes | Não dá para usar a classe original |
| 2 | **Todos os tabuleiros são finais** (o jogo já acabou). Não existe nenhum exemplo de "Tem jogo" | Uma das quatro classes não existe no dataset |
| 3 | A classe `negative` mistura vitórias de O e empates | É preciso separá-las |
| 4 | As classes são muito desbalanceadas depois da reclassificação: 626 X, 316 O e apenas **16 empates** | O Empate não alcança as 200 amostras sugeridas |
| 5 | Os valores são texto (`x`, `o`, `b`) | É preciso converter para número |

Não foram encontrados valores ausentes nem linhas duplicadas (0 duplicatas nas 958 linhas).

### 2.3 Passos executados e justificativas

1. **Conversão para número:** `x = 1`, `o = -1`, `b = 0` (vazio). A codificação com sinais opostos para X e O e zero para vazio preserva a simetria entre os jogadores e permite somar linhas para detectar vitórias.
2. **Reclassificação:** a classe original foi descartada. A função `avaliar_tabuleiro` verifica as 3 linhas, as 3 colunas e as 2 diagonais e devolve "Jogador X venceu", "Jogador O venceu", "Empate" (tabuleiro cheio sem vencedor) ou "Tem jogo". Isso resolve os problemas 1 e 3. Resultado sobre os 958 tabuleiros da UCI: **626 X, 316 O, 16 Empate**.
3. **Geração da classe "Tem jogo":** como a UCI só tem jogos terminados, foram simulados **200 tabuleiros em andamento** (`simular_jogos_em_andamento`). Cada simulação começa com o tabuleiro vazio, X joga primeiro, os jogadores alternam jogadas em casas vazias escolhidas ao acaso, e a partida para depois de 1 a 8 jogadas. Simulações que terminam com vencedor ou tabuleiro cheio, e tabuleiros repetidos, são descartados. Isso resolve o problema 2 e garante tabuleiros válidos (respeitam a ordem de jogadas).
4. **Balanceamento por amostragem:** foram sorteadas 200 amostras de cada uma das classes X, O e Tem jogo (`random_state=42`). Não foram usadas todas as instâncias, como o enunciado pede. Para Empate foram usados **todos os 16** exemplos, pois é o máximo que o jogo permite neste dataset.

### 2.4 Dataset final

| Classe | Amostras |
|---|---|
| Jogador X venceu | 200 |
| Jogador O venceu | 200 |
| Tem jogo | 200 |
| Empate | 16 |
| **Total** | **616** |

Peças no tabuleiro, por classe: Tem jogo tem de 1 a 8 (média 4,6). X venceu tem de 5 a 9 (média 6,7). O venceu tem de 6 a 8 (média 7,0). Empate tem sempre 9.

**Limitação assumida:** o Empate fica sub-representado (16 contra 200). Os efeitos disso nas métricas são discutidos na seção 6.

## 3. Pré-processamento

Foram testadas duas abordagens, como o enunciado pede.

### Abordagem 1: apenas o tabuleiro

A entrada são as 9 casas já convertidas (`1`, `-1`, `0`), que dão 9 features.

### Abordagem 2: features derivadas

São 7 features calculadas por `extrair_features`:

| Feature | Definição |
|---|---|
| `qtd_x` | Quantidade de X |
| `qtd_o` | Quantidade de O |
| `posicoes_ocupadas` | `qtd_x + qtd_o` |
| `linhas_2x` | Linhas, colunas e diagonais com 2 X e a terceira casa vazia |
| `linhas_2o` | Idem para O |
| `casas_vazias` | Quantidade de casas vazias |
| `jogador_vez` | `1` (X) se `qtd_x == qtd_o`, senão `-1` (O), pois X sempre começa |

Para contar `linhas_2x` e `linhas_2o`, só entram linhas em que a terceira casa está **vazia**, ou seja, em que o jogador realmente ameaça fechar uma linha.

### Normalização

Os dados são normalizados com `StandardScaler`, ajustado **somente no treino** para não vazar informação, para os algoritmos baseados em distância ou em gradiente (k-NN, MLP, K-Means). Árvore de Decisão e Random Forest usam os dados originais, pois dividem por limiares e não dependem da escala.

## 4. Divisão do dataset

A divisão é estratificada (mantém a proporção das classes) com `random_state=42`:

| Conjunto | Tamanho | X | O | Tem jogo | Empate |
|---|---|---|---|---|---|
| Treino (70%) | 430 | 140 | 140 | 140 | 10 |
| Validação (15%) | 93 | 30 | 30 | 30 | 3 |
| Teste (15%) | 93 | 30 | 30 | 30 | 3 |

- **Os mesmos conjuntos** são usados em todos os algoritmos.
- A **validação** serve para escolher os parâmetros de cada algoritmo. O **teste** só é usado depois, com os parâmetros já definidos, para medir o desempenho final.
- As duas abordagens usam as mesmas linhas, o que torna a comparação justa.
- Os conjuntos existem em memória (as variáveis `X1_train`, `X1_val`, `X1_test`, etc. em `jogo.py`). Eles ainda não são salvos em arquivos (pendência).

## 5. Algoritmos e parametrização

Todos os parâmetros foram escolhidos por **busca em grade** (todas as combinações) com a **maior acurácia no conjunto de validação**. Em caso de empate, vale a primeira combinação testada. As métricas de precision, recall e F-measure usam **média macro** (média simples das 4 classes), para que a classe pequena (Empate) pese o mesmo que as demais.

### 5.1 k-NN
Classifica um tabuleiro pela classe majoritária entre os `k` exemplos de treino mais próximos (distância euclidiana). Testados: `k` ímpar de 1 a 15. Ímpares evitam empate na votação. Usa dados normalizados.

### 5.2 MLP (rede neural)
Rede *feed-forward* treinada por retropropagação.

- **Grade:** camadas ocultas `(8)`, `(16)`, `(32)`, `(16,8)`, `(32,16)`; ativação `relu` ou `tanh`; regularização L2 `alpha` 0,0001 ou 0,01.
- `max_iter=5000`. Com 2000, várias configurações não convergiam.
- **Topologias escolhidas:**
  - Abordagem 1: 9 entradas → 1 camada oculta de 16 neurônios (ReLU) → 4 saídas.
  - Abordagem 2: 7 entradas → 1 camada oculta de 8 neurônios (ReLU) → 4 saídas.
- A saída tem 4 neurônios (softmax), um por classe.

### 5.3 Árvore de Decisão
Constrói perguntas do tipo "casa X ≤ valor?" que dividem os dados, escolhendo a divisão que mais reduz a impureza, até chegar a folhas com uma classe.

- **Grade:** `max_depth` 2, 3, 4, 5, 6, 8 ou sem limite; `min_samples_leaf` 1, 2, 5 ou 10; critério `gini` ou `entropy`. Limitar profundidade e folhas mínimas controla o overfitting.

### 5.4 Random Forest (algoritmo livre)
**Como funciona:** treina muitas árvores de decisão e a classe final é a mais votada entre elas. Cada árvore é treinada com uma amostra sorteada com reposição do treino (*bagging*), e em cada divisão considera só um subconjunto aleatório das features. As árvores ficam diferentes entre si, os erros individuais tendem a se compensar na votação, e isso reduz o overfitting de uma árvore isolada.

- **Grade:** `n_estimators` 50, 100 ou 200; `max_depth` 3, 5, 8 ou sem limite; `min_samples_leaf` 1, 2 ou 5.

### 5.5 K-Means (algoritmo livre)
**Como funciona:** o K-Means é **não supervisionado**. Ele escolhe `k` centroides, atribui cada ponto ao centroide mais próximo e recalcula os centroides como a média dos pontos de cada grupo, repetindo até estabilizar. Para usá-lo como classificador (classe `KMeansClassificador`), cada cluster recebe a **classe majoritária dos exemplos de treino** que caíram nele. Um tabuleiro novo recebe a classe do cluster mais próximo.

- **Grade:** `n_clusters` 4, 8, 16, 32, 64 ou 128. Usam-se mais clusters que classes, pois cada classe ocupa várias regiões do espaço.
- O número efetivo de clusters é limitado ao número de pontos distintos nos dados. Na Abordagem 2 existem só 42 pontos distintos (as features são discretas), então pedir 64 ou 128 clusters gera clusters vazios.
- Usa dados normalizados.

*(Um SVM foi testado numa versão anterior e substituído pelo K-Means.)*

### 5.6 Parâmetros escolhidos

| Algoritmo | Abordagem 1 (tabuleiro) | Abordagem 2 (features) |
|---|---|---|
| k-NN | k = 7 | k = 3 |
| MLP | (16), relu, alpha 0,0001 | (8), relu, alpha 0,0001 |
| Árvore de Decisão | sem limite de profundidade, folha mín. 1, entropy | profundidade 6, folha mín. 1, entropy |
| Random Forest | 50 árvores, sem limite de profundidade, folha mín. 2 | 50 árvores, profundidade 5, folha mín. 1 |
| K-Means | 64 clusters | 32 clusters |

## 6. Resultados

Resultados no conjunto de **teste** (93 amostras), com métricas macro. Tempos em segundos (variam um pouco a cada execução).

### 6.1 Abordagem 1: tabuleiro (9 features)

| Algoritmo | Acc. validação | Acc. treino | **Acurácia teste** | Precision | Recall | F-measure | Tempo treino |
|---|---|---|---|---|---|---|---|
| k-NN | 0,7957 | – | 0,6989 | 0,5284 | 0,5417 | 0,5284 | – |
| MLP | 0,8925 | 0,9465 | **0,8065** | 0,6019 | 0,6250 | 0,6125 | 0,72 |
| Árvore de Decisão | 0,7097 | 1,0000 | 0,7204 | 0,5478 | 0,5583 | 0,5505 | 0,001 |
| Random Forest | 0,8172 | 0,9907 | **0,8065** | 0,6073 | 0,6250 | 0,6116 | 0,04 |
| K-Means | 0,7097 | 0,7558 | 0,6237 | 0,4603 | 0,4833 | 0,4673 | 0,04 |

### 6.2 Abordagem 2: features derivadas (7 features)

| Algoritmo | Acc. validação | Acc. treino | **Acurácia teste** | Precision | Recall | F-measure | Tempo treino |
|---|---|---|---|---|---|---|---|
| k-NN | 0,8710 | – | 0,8817 | 0,6746 | 0,6833 | 0,6702 | – |
| MLP | 0,8710 | 0,9349 | **0,8925** | 0,6817 | 0,6917 | 0,6795 | 0,26 |
| Árvore de Decisão | 0,8710 | 0,9302 | 0,8602 | 0,6673 | 0,6667 | 0,6532 | 0,001 |
| Random Forest | 0,8710 | 0,9302 | 0,8495 | 0,6602 | 0,6583 | 0,6432 | 0,03 |
| K-Means | 0,8710 | 0,9302 | 0,8817 | 0,6746 | 0,6833 | 0,6702 | 0,02 |

Para o k-NN, a acurácia de treino e os tempos ainda não são coletados (a função do k-NN é anterior à função genérica). Essa é uma pendência.

### 6.3 Análise

**Abordagem 2 é melhor e mais barata.**
- A acurácia no teste sobe de 0,62–0,81 (Abordagem 1) para 0,85–0,89 (Abordagem 2) em todos os algoritmos.
- Ela usa 7 em vez de 9 features.
- O overfitting é menor. Na Abordagem 1 a Árvore chega a 1,00 de acurácia de treino contra 0,72 de teste, e o Random Forest a 0,99 contra 0,81. Na Abordagem 2 a diferença entre treino e teste fica em cerca de 0,04 a 0,08.
- Os tempos são pequenos em tudo (o MLP é o mais lento, cerca de 0,3 a 0,7 s). O custo extra da Abordagem 2 é só o cálculo das 7 features, que é desprezível.

**Por que as features ajudam:** contar peças, casas vazias e linhas ameaçadas entrega ao modelo informação sobre "quanto do jogo já foi jogado" e sobre ameaças. O tabuleiro cru exige que o algoritmo descubra sozinho os padrões de três em linha a partir de 9 números, com só 430 exemplos de treino.

**Os algoritmos ficaram empatados na Abordagem 2.** A validação deu 0,8710 para os cinco, e no teste a diferença entre o melhor (MLP, 0,8925) e o pior (Random Forest, 0,8495) é de 4 pontos percentuais. Em 93 amostras isso equivale a cerca de 4 tabuleiros, o que está dentro do ruído de um conjunto de teste tão pequeno. Não dá para afirmar com segurança qual algoritmo é superior.

**A classe Empate é o principal limitador das métricas.**
- O teste tem só 3 empates. O recall e a precision macro sobem a cerca de 0,67 a 0,69, o que é compatível com um modelo que acerta bem 3 classes e erra o Empate.
- Na Abordagem 2, um empate (tabuleiro cheio, 5 X e 4 O) tem as mesmas 7 features de uma vitória de X com tabuleiro cheio, então as features **não conseguem separá-los**. É uma limitação das features escolhidas e não do algoritmo.
- Existem apenas 42 pontos distintos no espaço de features da Abordagem 2, então muitos exemplos de teste repetem exemplos de treino. Isso pode inflar a acurácia da Abordagem 2.

### 6.4 Escolha do melhor algoritmo

*(a ser fechada na conclusão final)* Pelos resultados acima, a **Abordagem 2** é a escolhida. Entre os algoritmos, o **MLP** teve a maior acurácia (0,8925) e F-measure (0,6795), e foi o melhor também na Abordagem 1 (empatado com o Random Forest em acurácia). A diferença para os demais é pequena e dentro do ruído, então a decisão final deve levar em conta também o custo e a interpretabilidade: a Árvore de Decisão é a mais rápida e interpretável (treino de cerca de 1 ms, com 0,86 de acurácia). A escolha definitiva será confirmada depois de resolver o Empate (seção 8) e de reavaliar com os conjuntos salvos.

## 7. Ferramentas de IA utilizadas

- **Claude Code (Anthropic, modelo Claude Sonnet 5.5):** usado para ajudar a entender o trabalho, tirar dúvidas, estruturar e corrigir o relatório.
- **Gamma IA:** usado para gerar o modelo dos slides para a apresentação do trabalho.

## 8. Pendências

- [ ] **Front end:** jogo humano × máquina aleatória, com a IA indicando a cada jogada se há vitória, empate ou jogo em andamento. Deve contar acertos e erros e medir a acurácia durante as interações. Se a IA não detectar o fim, o jogo encerra; se detectar o fim incorretamente, continua.
- [ ] **Gráficos comparativos** dos algoritmos e abordagens.
- [ ] **Salvar os conjuntos** de treino, validação e teste em arquivos (CSV).
- [ ] **Tratar o Empate:** gerar mais empates sintéticos ou rever as features, que hoje não separam Empate de X com tabuleiro cheio.
- [ ] **Coletar acurácia de treino e tempos do k-NN** com a função genérica.
- [ ] **Resultados do front end** (acurácia da IA nas interações reais).
- [ ] **Relatório em PPT**, **vídeo de até 10 min** com todos os integrantes falando e **conclusão** (dificuldades e ganhos).
- [ ] Atualizar o `resultados.txt`, que ainda mostra a execução antiga (só k-NN).
