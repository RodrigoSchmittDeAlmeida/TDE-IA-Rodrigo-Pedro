import pandas as pd
import numpy as np
import random
import tkinter as tk
from tkinter import messagebox

random.seed(42)  # fixa a simulacao de "Tem jogo" para os resultados serem reproduziveis
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



# OUTROS ALGORITMOS (MLP, Arvore de Decisao, K-Means, Random Forest)

import time
from itertools import product
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

#k-means e nao supervisionado, entao usamos como classificador: agrupamos o treino em clusters
#e cada cluster recebe o rotulo da classe majoritaria dos seus exemplos de treino
class KMeansClassificador(BaseEstimator, ClassifierMixin):
    def __init__(self, n_clusters=8, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        #nao da para ter mais clusters do que pontos distintos (features discretas repetem muito)
        n_efetivo = min(self.n_clusters, len(np.unique(X, axis=0)))
        self.kmeans_ = KMeans(n_clusters=n_efetivo, n_init=10, random_state=self.random_state)
        clusters = self.kmeans_.fit_predict(X)
        self.classes_ = np.unique(y)
        self.rotulos_ = {}
        for c in range(n_efetivo):
            rotulos, contagens = np.unique(y[clusters == c], return_counts=True)
            self.rotulos_[c] = rotulos[np.argmax(contagens)] if len(rotulos) > 0 else self.classes_[0]
        return self

    def predict(self, X):
        return np.array([self.rotulos_[c] for c in self.kmeans_.predict(X)])


#funcao generica: testa todas as combinacoes de parametros na validacao e avalia a melhor no teste
def avaliar_modelo(nome, construtor, grade, X_tr, y_tr, X_v, y_v, X_te, y_te, nome_abordagem):
    nomes_params = list(grade.keys())
    melhor_params = None
    melhor_acuracia_val = -1

    for valores in product(*grade.values()):
        params = dict(zip(nomes_params, valores))
        modelo_temp = construtor(**params)
        modelo_temp.fit(X_tr, y_tr)
        acc_val = accuracy_score(y_v, modelo_temp.predict(X_v))

        if acc_val > melhor_acuracia_val:
            melhor_acuracia_val = acc_val
            melhor_params = params

    #treina o modelo final com os melhores parametros e mede tempos de treino e predicao (custo)
    modelo_final = construtor(**melhor_params)
    inicio = time.time()
    modelo_final.fit(X_tr, y_tr)
    tempo_treino = time.time() - inicio

    inicio = time.time()
    y_pred_test = modelo_final.predict(X_te)
    tempo_predicao = time.time() - inicio

    acc_treino = accuracy_score(y_tr, modelo_final.predict(X_tr))
    acc = accuracy_score(y_te, y_pred_test)
    prec = precision_score(y_te, y_pred_test, average='macro', zero_division=0)
    rec = recall_score(y_te, y_pred_test, average='macro', zero_division=0)
    f1 = f1_score(y_te, y_pred_test, average='macro', zero_division=0)

    print(f"--- {nome} | {nome_abordagem} ---")
    print(f"Melhores parâmetros na Validação: {melhor_params} (Acurácia: {melhor_acuracia_val:.4f})")
    print(f"Desempenho no Teste final:")
    print(f"Acurácia treino: {acc_treino:.4f} | Acurácia teste: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F-Measure: {f1:.4f}")
    print(f"Tempo treino: {tempo_treino:.4f}s | Tempo predição: {tempo_predicao:.4f}s\n")

    return modelo_final, {'algoritmo': nome, 'abordagem': nome_abordagem, 'parametros': melhor_params,
                          'acc_val': melhor_acuracia_val, 'acc_treino': acc_treino, 'acc_teste': acc,
                          'precision': prec, 'recall': rec, 'f1': f1,
                          'tempo_treino': tempo_treino, 'tempo_predicao': tempo_predicao}


# grades de parametros testados na validacao
grade_mlp = {
    'hidden_layer_sizes': [(8,), (16,), (32,), (16, 8), (32, 16)],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.01],
    'max_iter': [5000],
    'random_state': [42],
}
grade_arvore = {
    'max_depth': [2, 3, 4, 5, 6, 8, None],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'random_state': [42],
}
#mais clusters do que classes, pois cada classe pode ocupar varias regioes do espaco
grade_kmeans = {
    'n_clusters': [4, 8, 16, 32, 64, 128],
    'random_state': [42],
}
grade_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 8, None],
    'min_samples_leaf': [1, 2, 5],
    'random_state': [42],
}

#dados por abordagem (MLP e K-Means usam os dados normalizados; arvore e RF nao precisam de normalizacao)
abordagens = {
    'Abordagem 1 (Tabuleiro)': {
        'normalizado': (X1_train_scaled, X1_val_scaled, X1_test_scaled),
        'original': (X1_train, X1_val, X1_test),
    },
    'Abordagem 2 (Features Extraídas)': {
        'normalizado': (X2_train_scaled, X2_val_scaled, X2_test_scaled),
        'original': (X2_train, X2_val, X2_test),
    },
}

algoritmos = [
    ('MLP', MLPClassifier, grade_mlp, 'normalizado'),
    ('Árvore de Decisão', DecisionTreeClassifier, grade_arvore, 'original'),
    ('K-Means', KMeansClassificador, grade_kmeans, 'normalizado'),
    ('Random Forest', RandomForestClassifier, grade_rf, 'original'),
]

resultados_modelos = []
modelos_treinados = {}
for nome_ab, dados in abordagens.items():
    for nome_alg, construtor, grade, tipo_dado in algoritmos:
        X_tr, X_v, X_te = dados[tipo_dado]
        modelo, resultado = avaliar_modelo(nome_alg, construtor, grade, X_tr, y_train, X_v, y_val, X_te, y_test, nome_ab)
        resultados_modelos.append(resultado)
        modelos_treinados[(nome_alg, nome_ab)] = modelo

class JogoDaVelhaGUI:
    def __init__(self, master, modelo_ia, scaler, colunas_features):
        self.master = master
        self.master.title("Tic-Tac-Toe IA - T1")
        self.master.geometry("400x600")
        
        self.modelo_ia = modelo_ia
        self.scaler = scaler
        self.colunas_features = colunas_features
        
        self.tab = [0] * 9
        self.botoes = []
        
        self.acertos_ia = 0
        self.total_predicoes = 0
        
        # Trava para impedir cliques rápidos do humano
        self.bloqueado = False 
        
        # Frame do Tabuleiro
        frame_tab = tk.Frame(master)
        frame_tab.pack(pady=10)
        
        for i in range(9):
            btn = tk.Button(frame_tab, text="", font=('Helvetica', 24, 'bold'), width=5, height=2,
                            command=lambda i=i: self.jogada_humano(i))
            btn.grid(row=i//3, column=i%3, padx=5, pady=5)
            self.botoes.append(btn)
            
        # Frame de Informações e Score
        frame_status = tk.Frame(master)
        frame_status.pack(pady=10)
        
        self.lbl_status = tk.Label(frame_status, text="Sua vez! Jogue com o X", font=('Helvetica', 12, 'bold'))
        self.lbl_status.pack()
        
        self.lbl_ia = tk.Label(frame_status, text="Previsão da IA: Aguardando...", font=('Helvetica', 11), fg="blue")
        self.lbl_ia.pack(pady=5)
        
        self.lbl_score = tk.Label(frame_status, text="Acurácia da IA: 0.00%", font=('Helvetica', 11))
        self.lbl_score.pack(pady=5)

        # Botão de Reiniciar
        self.btn_reiniciar = tk.Button(frame_status, text="Reiniciar Partida", font=('Helvetica', 10, 'bold'), 
                                       command=self.reiniciar_jogo, state="disabled")
        self.btn_reiniciar.pack(pady=10)

    def reiniciar_jogo(self):
        self.tab = [0] * 9
        self.bloqueado = False # Libera a interface ao reiniciar
        
        for btn in self.botoes:
            btn.config(text="", state="normal")
        
        self.lbl_status.config(text="Sua vez! Jogue com o X")
        self.lbl_ia.config(text="Previsão da IA: Aguardando...")
        self.btn_reiniciar.config(state="disabled")

    def jogada_humano(self, idx):
        # Se estiver bloqueado ou a casa ocupada, ignora o clique
        if self.bloqueado or self.tab[idx] != 0:
            return
        
        # Bloqueia cliques imediatamente
        self.bloqueado = True 
        
        self.tab[idx] = 1
        self.botoes[idx].config(text="X", state="disabled", disabledforeground="black")
        self.master.update_idletasks()
        
        if self.processar_turno():
            self.lbl_status.config(text="Máquina pensando...")
            self.master.after(500, self.jogada_maquina)

    def jogada_maquina(self):
        posicoes_vazias = [i for i, v in enumerate(self.tab) if v == 0]
        if not posicoes_vazias:
            return
            
        jogada_escolhida = None

        # # 1. Inteligência da Máquina: Tenta ganhar
        # for pos in posicoes_vazias:
        #     self.tab[pos] = -1
        #     if avaliar_tabuleiro(self.tab) == 'Jogador O venceu':
        #         jogada_escolhida = pos
        #     self.tab[pos] = 0
        #     if jogada_escolhida is not None:
        #         break

        # # 2. Inteligência da Máquina: Tenta bloquear o X
        # if jogada_escolhida is None:
        #     for pos in posicoes_vazias:
        #         self.tab[pos] = 1
        #         if avaliar_tabuleiro(self.tab) == 'Jogador X venceu':
        #             jogada_escolhida = pos
        #         self.tab[pos] = 0
        #         if jogada_escolhida is not None:
        #             break

        # 3. Inteligência da Máquina: Joga aleatório
        if jogada_escolhida is None:
            jogada_escolhida = random.choice(posicoes_vazias)

        self.tab[jogada_escolhida] = -1
        self.botoes[jogada_escolhida].config(text="O", state="disabled", disabledforeground="red")
        self.master.update_idletasks()
        
        self.lbl_status.config(text="Sua vez! Jogue com o X")
        
        # Só libera o botão para o humano se o jogo não tiver acabado
        if self.processar_turno():
            self.bloqueado = False

    def processar_turno(self):
        features = extrair_features(self.tab)
        df_instancia = pd.DataFrame([features.values], columns=self.colunas_features)
        features_norm = self.scaler.transform(df_instancia)
        
        estado_ia = self.modelo_ia.predict(features_norm)[0]
        estado_real = avaliar_tabuleiro(self.tab)
        
        self.total_predicoes += 1
        if estado_ia == estado_real:
            self.acertos_ia += 1
            
        acc = (self.acertos_ia / self.total_predicoes) * 100
        
        self.lbl_ia.config(text=f"Previsão da IA: {estado_ia}\nEstado Real: {estado_real}")
        self.lbl_score.config(text=f"Acurácia da IA: {acc:.2f}% ({self.acertos_ia}/{self.total_predicoes})")
        
        if estado_ia == 'Tem jogo' and estado_real != 'Tem jogo':
            messagebox.showwarning("Erro da IA", f"REGRA DO ENUNCIADO:\n\nA IA não detectou o fim de jogo (previu 'Tem jogo').\nRealidade: {estado_real}\n\nEncerrando a partida prematuramente.")
            self.desativar_botoes()
            return False
            
        if estado_ia != 'Tem jogo' and estado_real == 'Tem jogo':
            messagebox.showinfo("Falso Fim de Jogo", f"REGRA DO ENUNCIADO:\n\nA IA detectou incorretamente o fim de jogo (previu '{estado_ia}').\nRealidade: O jogo não acabou.\n\nA partida continuará normalmente.")
            return True
            
        if estado_real != 'Tem jogo':
            messagebox.showinfo("Fim de Jogo", f"O jogo acabou.\n\nResultado final correto: {estado_real}\nPrevisão da IA: {estado_ia}")
            self.desativar_botoes()
            return False
            
        return True

    def desativar_botoes(self):
        self.bloqueado = True
        for btn in self.botoes:
            btn.config(state="disabled")
        self.btn_reiniciar.config(state="normal")
        
        # Inicialização do Front-End
if __name__ == "__main__":
    # Garante que o modelo escolhido será a MLP com a Abordagem 2
    modelo_escolhido = modelos_treinados[('MLP', 'Abordagem 2 (Features Extraídas)')]
    
    janela_principal = tk.Tk()
    app = JogoDaVelhaGUI(janela_principal, modelo_escolhido, scaler2, colunas_features)
    janela_principal.mainloop()