import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


#função para avaliar o estado de um tabuleiro 3x3
def avaliar_tabuleiro(linha):
    #transforma a linha do pandas em uma matriz 3x3 para facilitar a checagem
    tab = np.array(linha[:9]).reshape(3, 3)

    #funcao auxiliar para checar se alguem completou uma linha, coluna, diagonal

    def checar_vencedor(jogador):
        #checa as linhas e colunas
        for i in range(3):
            if all(tab[i, :] == jogador) or all(tab[:, i] == jogador):
                return True
        #checa as diagonais
        if all(np.diag(tab) == jogador) or all(np.diag(np.fliplr(tab)) == jogador):
            return True
        return False

    if checar_vencedor(1): # 1 representa o 'X'
        return "Jogador X venceu"
    elif checar_vencedor(-1): # -1 representa o "O"
        return "Jogador O venceu"
    elif not np.any(tab == 0): # 0 representa vazio ('b'). Se nao tem vazio ninguem venceu, empatou
        return 'Empate'
    else:
        return 'Tem jogo'


#  carregar o dataset
colunas = ['TL', 'TM', 'TR', 'ML', 'MM', 'MR', 'BL', 'BM', 'BR', 'class_original']
df_uci = pd.read_csv('tic-tac-toe.data.csv', names=colunas, header=None)

#conversao simbolico-numero (pré-processamento)
mapa_valores = {'x': 1, 'o': -1, 'b': 0}
for col in colunas[:-1]:
    df_uci[col] = df_uci[col].map(mapa_valores)

# aplica nossa propria classificacao para ignorar a classe original e gerar as 4 classes 
df_uci['Classe_Alvo'] = df_uci.apply(avaliar_tabuleiro, axis=1)

#descartar a coluna de classe original da ICU, pois não precisamos mais dela
df_uci = df_uci.drop('class_original', axis=1)

# geracao de tabuleiros sinteticos para a classe "tem jogo"
def simular_jogos_em_andamento(qtd_desejada):
    jogos_em_andamento = []

    while len(jogos_em_andamento) < qtd_desejada:
        #inicia tabuleiro vazio
        tab = [0] * 9
        #decide aleatoriamente em qual turno parar entre o turno 1 e 8
        turnos_para_jogar = random.randint(1, 8)
        jogador_atual = 1 # 'x' sempre começa

        for _ in range(turnos_para_jogar):
            posicoes_vazias = [i for i, v in enumerate(tab) if v == 0]
            if not posicoes_vazias: break

            jogada = random.choice(posicoes_vazias)
            tab[jogada] = jogador_atual

            #se alguem ganhou acidentalmente no meio da simulacao, paramos e descartamos
            if avaliar_tabuleiro(tab) != 'Tem jogo':
                break

            #troca o jogador
            jogador_atual = -1 if jogador_atual == 1 else 1

        #se apos os turnos o status for estrutamente "tem jogo", salvamos
        if avaliar_tabuleiro(tab) == 'Tem jogo':
            #evita duplicatas adicionando apenas se não existir
            if tab not in jogos_em_andamento:
                jogos_em_andamento.append(tab)

    return jogos_em_andamento

amostras_tem_jogo = simular_jogos_em_andamento(200)
df_tem_jogo = pd.DataFrame(amostras_tem_jogo, columns=colunas[:-1])
df_tem_jogo['Classe_Alvo'] = 'Tem jogo'

# unir e balancear o dataset final
df_final = pd.concat([df_uci, df_tem_jogo], ignore_index=True)

#balanceamento (amostragem de 200 de cada, exceto Empate que é limitado pela matematica do jogo)
df_balanceado = pd.concat([
    df_final[df_final['Classe_Alvo'] == 'Jogador X venceu'].sample(200, random_state=42),
    df_final[df_final['Classe_Alvo'] == 'Jogador O venceu'].sample(n=min(200, len(df_final[df_final['Classe_Alvo'] == 'Jogador O venceu'])), random_state=42),
    df_final[df_final['Classe_Alvo'] == 'Empate'], # Usamos todos os empates possíveis (geralmente 16)
    df_final[df_final['Classe_Alvo'] == 'Tem jogo'].sample(200, random_state=42)
])

print("Distribuição das classes no Dataset Final:")
print(df_balanceado['Classe_Alvo'].value_counts())


#abordagem 2

def extrair_features(linha):
    #transforma as primeira 9 colunas no tabuleiro 3x3
    tab = np.array(linha[:9], dtype=int).reshape(3,3)

    qtd_x = np.sum(tab == 1)
    qtd_o = np.sum(tab == -1)
    casas_vazias = np.sum(tab == 0)
    posicoes_ocupadas = qtd_x + qtd_o

    #define de quem é a vez (x sempre começa, entao se a quantidade for igual, é a vez do X)
    jogador_vez = 1 if qtd_x == qtd_o else -1

    def contar_linhas_duplas(jogador):
        alvo = jogador*2
        duplas = 0
        #checa as 3 linhas e 3 colunas
        for i in range(3):
            if np.sum(tab[i, :]) == alvo and np.sum(tab[i, :] == 0) == 1:
                duplas += 1
            if np.sum(tab[:, i]) == alvo and np.sum(tab[:, i] == 0) ==1:
                duplas += 1

        #checa as 2 diagonais
        diag1 = np.diag(tab)
        diag2 = np.diag(np.fliplr(tab))
        if np.sum(diag1) == alvo and np.sum(diag1 == 0) == 1:
            duplas += 1
        if np.sum(diag2) == alvo and np.sum(diag2 == 0) == 1:
            duplas += 1
        return duplas

    linhas_2x = contar_linhas_duplas(1)
    linhas_2o = contar_linhas_duplas(-1)

    return pd.Series([qtd_x, qtd_o, posicoes_ocupadas, linhas_2x, linhas_2o, casas_vazias, jogador_vez])

# Nomes das novas colunas de entrada exigidas pelo trabalho
colunas_features = ['qtd_x', 'qtd_o', 'posicoes_ocupadas', 'linhas_2x', 'linhas_2o', 'casas_vazias', 'jogador_vez']

# Aplica a extração para criar o dataset da Abordagem 2
df_abordagem2 = df_balanceado.copy()
df_abordagem2[colunas_features] = df_abordagem2.apply(extrair_features, axis=1)


#DIVISAO DE TREINO, VALIDACAO E TESTE

#separando os alvos(y)
y = df_balanceado['Classe_Alvo']

# X para abordagem 1 (as 9 casas do tabuleiro)
X1 = df_balanceado.iloc[:, :9]

# X para abordagem 2 (as 7 features matemáticas)
X2 = df_abordagem2[colunas_features]

# Função auxiliar para dividir em 70% Treino, 15% Validação, 15% Teste
def dividir_dados(X, y):
    # Primeiro isolamos 15% para o Teste Final
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    # Sobram 85%. Desses, tiramos uma fatia equivalente a 15% do total geral para Validação
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test


# Obtendo os conjuntos fisicamente separados para cada abordagem
X1_train, X1_val, X1_test, y_train, y_val, y_test = dividir_dados(X1, y)
X2_train, X2_val, X2_test, _, _, _ = dividir_dados(X2, y)



# ALGORITMO K-nn

# normalizacao dos dados (essencial para algoritmos baseados em distancia)
scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_val_scaled = scaler1.transform(X1_val)
X1_test_scaled = scaler1.transform(X1_test)

scaler2 = StandardScaler()
X2_train_scaled = scaler2.fit_transform(X2_train)
X2_val_scaled = scaler2.transform(X2_val)
X2_test_scaled = scaler2.transform(X2_test)

#funcao para treinar, validar parametros e testar o k-NN
def avaliar_knn(X_tr, y_tr, X_v, y_v, X_te, y_te, nome_abordagem):
    melhor_k = 1
    melhor_acuracia_val = 0

    #testando valores impares para k (de 1 a 15) 
    for k in range (1, 16, 2):
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_tr, y_tr)

        #validando o parametro
        y_pred_val = knn_temp.predict(X_v)
        acc_val = accuracy_score(y_v, y_pred_val)

        if acc_val > melhor_acuracia_val:
            melhor_acuracia_val = acc_val
            melhor_k = k

    print(f"--- {nome_abordagem} ---")
    print(f"Melhor parâmetro encontrado na Validação: k = {melhor_k} (Acurácia: {melhor_acuracia_val:.4f})")

    #teste definitivo com melhor hiperparametro encontrado

    knn_final = KNeighborsClassifier(n_neighbors=melhor_k)
    knn_final.fit(X_tr, y_tr)
    y_pred_test = knn_final.predict(X_te)

    #extracao das metricas (usando 'macro' para calcular a média equilibrada entre as 4 classes)
    acc = accuracy_score(y_te, y_pred_test)
    prec = precision_score(y_te, y_pred_test, average='macro',zero_division = 0)
    rec = recall_score(y_te, y_pred_test, average='macro', zero_division=0)
    f1 = f1_score(y_te, y_pred_test, average='macro', zero_division=0)

    print(f"Desempenho no Teste final:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F-Measure: {f1:.4f}\n")

    return knn_final

# Executando para a Abordagem 1 (Apenas o tabuleiro)
modelo_knn_ab1 = avaliar_knn(X1_train_scaled, y_train, X1_val_scaled, y_val, X1_test_scaled, y_test, "Abordagem 1 (Tabuleiro)")

# Executando para a Abordagem 2 (Features estruturadas)
modelo_knn_ab2 = avaliar_knn(X2_train_scaled, y_train, X2_val_scaled, y_val, X2_test_scaled, y_test, "Abordagem 2 (Features Extraídas)")
