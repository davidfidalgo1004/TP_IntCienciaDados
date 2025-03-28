# -*- coding: utf-8 -*-
"""
Script completo para análise do dataset "Global Water Consumption Dataset (2000-2024)".
Este script realiza as seguintes tarefas, conforme solicitado para o Trabalho Experimental 1:
  1. Carrega o ficheiro CSV completo, exibe informações iniciais e cria um novo DataFrame
     filtrado por países específicos (Italy, Japan, Spain, UK, USA), guardando-o num novo CSV.
  2. Plota a evolução do "Total Water Consumption" ao longo dos anos para os países selecionados.
  3. Cria um gráfico circular (pie chart) mostrando a distribuição percentual dos usos de água
     (agrícola, industrial e doméstico) para Spain no ano de 2020.
  4. Define uma função que retorna o ano e o valor da menor percentagem de "Agricultural Water Use"
     para um país fornecido como entrada.
  5. Plota um gráfico de dispersão com uma linha de regressão linear entre "Industrial Water Use" e
     "Groundwater Depletion Rate", utilizando o dataset completo, e interpreta a regressão.
  6. Utiliza técnicas de Machine Learning (Regressão Linear) para prever "Per Capita Water Use"
     com base nas variáveis "Country", "Agricultural Water Use" e "Rainfall Impact", documentando
     o processo e avaliando os resultados.

Certifique-se de ter os pacotes pandas, matplotlib, numpy e scikit-learn instalados.
Para instalar, pode usar: pip install pandas matplotlib numpy scikit-learn
"""

# Importação das bibliotecas necessárias
import pandas as pd  # Para manipulação e análise de dados (DataFrames)
import matplotlib.pyplot as plt  # Para criação de gráficos
import numpy as np  # Para operações numéricas, especialmente com arrays
from sklearn.model_selection import train_test_split  # Para dividir dados em treino e teste (ML)
from sklearn.linear_model import LinearRegression  # Para criar o modelo de Regressão Linear (ML)
from sklearn.metrics import mean_squared_error, r2_score  # Para avaliar o modelo de ML

# --- Tarefa 1: Carregar, Filtrar e Guardar Dados ---

# Carregar o dataset completo a partir do ficheiro CSV para um DataFrame do pandas.
# O ficheiro "cleaned_global_water_consumption.csv" deve estar no mesmo diretório que o script,
# ou o caminho completo deve ser fornecido.
df = pd.read_csv("cleaned_global_water_consumption.csv")

# Exibir informações básicas sobre o DataFrame carregado:
# - df.shape: Retorna um tuple com o número de linhas e colunas.
# - df.columns.tolist(): Lista os nomes de todas as colunas.
# - df.head(): Mostra as primeiras 5 linhas do DataFrame por defeito.
print("--- Análise Exploratória Inicial do Dataset Completo ---")
print("Dimensões (linhas, colunas):", df.shape)
print("Nomes das colunas:", df.columns.tolist())
print("\nPrimeiras 5 linhas:")
print(df.head())

# Exibir estatísticas descritivas para as colunas numéricas (contagem, média, desvio padrão, min, max, quartis).
print("\nEstatísticas Descritivas:")
print(df.describe())

# Verificar a existência de valores nulos (ausentes) em cada coluna.
# isnull() retorna um DataFrame booleano (True onde há nulo) e sum() conta os True por coluna.
print("\nContagem de Valores Nulos por Coluna:")
print(df.isnull().sum())
print("----------------------------------------------------")

# Definir a lista de países de interesse.
paises = ['Italy', 'Japan', 'Spain', 'UK', 'USA']

# Filtrar o DataFrame original ('df') para manter apenas as linhas onde a coluna 'Country'
# corresponde a um dos países na lista 'paises'.
df_filtrado = df[df['Country'].isin(paises)].copy() # Usar .copy() para evitar SettingWithCopyWarning

# Guardar o DataFrame filtrado num novo ficheiro CSV.
# index=False evita que o índice do DataFrame seja escrito como uma coluna no CSV.
df_filtrado.to_csv("filtered_global_water_consumption.csv", index=False)
print(f"--- Tarefa 1 Concluída ---")
print(f"Dados filtrados para os países {', '.join(paises)} guardados em 'filtered_global_water_consumption.csv'.")
print("----------------------------------------------------")


# --- Tarefa 2: Gráfico da Evolução do Consumo Total de Água ---

# É boa prática garantir que a coluna 'Year' é do tipo numérico para o plot.
# errors='coerce' transforma valores que não podem ser convertidos em NaN (Not a Number).
# A linha seguinte pode gerar um SettingWithCopyWarning se .copy() não foi usado acima.
# Como usamos .copy() na criação de df_filtrado, este aviso é evitado.
df_filtrado['Year'] = pd.to_numeric(df_filtrado['Year'], errors='coerce')

# Criar a figura e os eixos para o gráfico com um tamanho específico (largura 10, altura 6).
plt.figure(figsize=(12, 7)) # Aumentado ligeiramente para melhor visualização

# Iterar sobre cada país na lista 'paises'.
for pais in paises:
    # Filtrar o DataFrame 'df_filtrado' para obter os dados apenas do país atual.
    df_pais = df_filtrado[df_filtrado['Country'] == pais]
    # Plotar a evolução: 'Year' no eixo X, 'Total Water Consumption' no eixo Y.
    # marker='o' adiciona um marcador circular em cada ponto de dados.
    # label=pais define o nome que aparecerá na legenda para esta linha.
    plt.plot(df_pais['Year'], df_pais['Total Water Consumption (Billion Cubic Meters)'], marker='o', linestyle='-', label=pais)

# Adicionar rótulos aos eixos X e Y e um título ao gráfico.
plt.xlabel("Ano")
plt.ylabel("Consumo Total de Água (Bilhões de Metros Cúbicos)")
plt.title("Evolução Anual do Consumo Total de Água (2000-2024) por País Selecionado")

# Adicionar uma legenda para identificar as linhas de cada país.
plt.legend()

# Adicionar uma grelha ao fundo do gráfico para facilitar a leitura dos valores.
plt.grid(True)

# Ajustar o layout para evitar que os rótulos se sobreponham.
plt.tight_layout()

# Exibir o gráfico gerado.
print(f"--- Tarefa 2 Concluída ---")
print("A exibir o gráfico da evolução do consumo total de água...")
plt.show()
print("----------------------------------------------------")


# --- Tarefa 3: Gráfico Circular (Pie Chart) para Spain em 2020 ---

# Utiliza-se um bloco try-except para lidar com o caso de não existirem dados para Spain em 2020.
try:
    # Filtrar o DataFrame original ('df') para encontrar a linha correspondente a 'Spain' e 'Year' 2020.
    # .iloc[0] seleciona a primeira (e única esperada) linha que corresponde aos critérios.
    dados_spain_2020 = df[(df['Country'] == 'Spain') & (df['Year'] == 2020)].iloc[0]

    # Definir os rótulos (labels) para as fatias do gráfico circular.
    labels = ['Uso Agrícola (%)', 'Uso Industrial (%)', 'Uso Doméstico (%)']
    # Extrair os valores correspondentes do DataFrame filtrado.
    valores = [
        dados_spain_2020['Agricultural Water Use (%)'],
        dados_spain_2020['Industrial Water Use (%)'],
        dados_spain_2020['Household Water Use (%)']
    ]

    # Criar a figura para o gráfico circular com um tamanho específico (6x6).
    plt.figure(figsize=(7, 7)) # Ligeiramente maior para clareza
    # Criar o gráfico circular:
    # - valores: os dados a serem representados.
    # - labels: os rótulos para cada fatia.
    # - autopct='%1.1f%%': formata a percentagem a ser exibida em cada fatia (uma casa decimal).
    # - startangle=140: define o ângulo inicial da primeira fatia.
    plt.pie(valores, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'}) # Adicionado contorno

    # Adicionar um título ao gráfico.
    plt.title("Distribuição Percentual do Uso de Água em Espanha (2020)")

    # Exibir o gráfico.
    print(f"--- Tarefa 3 Concluída ---")
    print("A exibir o gráfico circular da distribuição do uso de água em Espanha (2020)...")
    plt.show()

# Se a linha para Spain em 2020 não for encontrada (IndexError), imprime uma mensagem.
except IndexError:
    print(f"--- Tarefa 3 Falhou ---")
    print("Não foram encontrados dados para 'Spain' no ano 2020 no dataset.")
print("----------------------------------------------------")


# --- Tarefa 4: Função para Encontrar o Menor Uso Agrícola por País ---

def menor_uso_agricola(df_completo, nome_pais):
    """
    Encontra e retorna o ano e o valor da menor percentagem de 'Agricultural Water Use'
    para um país específico no DataFrame fornecido.

    Parâmetros:
      df_completo (pd.DataFrame): O DataFrame completo contendo os dados.
      nome_pais (str): O nome do país a ser pesquisado (sensível a maiúsculas/minúsculas).

    Retorna:
      tuple: (ano_min, valor_min) - O ano e o valor mínimo encontrado.
             Retorna (None, None) se o país não for encontrado no DataFrame.
    """
    # Filtrar o DataFrame para obter apenas os dados do país especificado.
    df_country = df_completo[df_completo['Country'] == nome_pais]

    # Verificar se o DataFrame filtrado está vazio (país não encontrado).
    if df_country.empty:
        print(f"Aviso: Não foram encontrados dados para o país '{nome_pais}'.")
        return None, None # Retorna None para indicar que não encontrou

    # Encontrar o índice da linha onde 'Agricultural Water Use (%)' tem o valor mínimo.
    idx_min = df_country['Agricultural Water Use (%)'].idxmin()

    # Localizar a linha completa correspondente a esse índice mínimo.
    linha_min = df_country.loc[idx_min]

    # Extrair o ano ('Year') e o valor mínimo ('Agricultural Water Use (%)') dessa linha.
    ano_min = linha_min['Year']
    valor_min = linha_min['Agricultural Water Use (%)']

    # Retornar o ano e o valor mínimo.
    return int(ano_min), valor_min # Converte ano para inteiro para apresentação

# Exemplo de utilização da função para o país 'Italy'.
print(f"--- Tarefa 4: Teste da Função ---")
pais_exemplo = "Italy"
# Chama a função com o DataFrame completo 'df' e o nome do país.
ano, valor = menor_uso_agricola(df, pais_exemplo)

# Verifica se a função retornou valores válidos (não None).
if ano is not None:
    print(f"Para o país '{pais_exemplo}', o menor valor de 'Agricultural Water Use' ({valor:.2f}%) ocorreu no ano {ano}.")
else:
    # Mensagem caso o país não tenha sido encontrado (já impressa dentro da função).
    pass
print("----------------------------------------------------")


# --- Tarefa 5: Gráfico de Dispersão e Regressão Linear ---

# Selecionar as colunas relevantes ('Industrial Water Use (%)', 'Groundwater Depletion Rate (%)') do DataFrame completo.
# .dropna() remove as linhas onde qualquer um destes valores seja nulo, para evitar erros na regressão.
df_scatter = df[['Industrial Water Use (%)', 'Groundwater Depletion Rate (%)']].dropna()

# Criar a figura e os eixos para o gráfico de dispersão.
plt.figure(figsize=(10, 6))

# Criar o gráfico de dispersão:
# - Eixo X: 'Industrial Water Use (%)'.
# - Eixo Y: 'Groundwater Depletion Rate (%)'.
# - alpha=0.6: Define a transparência dos pontos (útil se houver sobreposição).
# - label='Dados': Rótulo para a legenda.
plt.scatter(df_scatter['Industrial Water Use (%)'], df_scatter['Groundwater Depletion Rate (%)'], alpha=0.6, label='Dados Observados')

# --- Cálculo e Plot da Regressão Linear ---
# Extrair os valores das colunas como arrays NumPy para a função polyfit.
x = df_scatter['Industrial Water Use (%)'].values
y = df_scatter['Groundwater Depletion Rate (%)'].values

# Calcular os coeficientes da regressão linear (polinómio de grau 1).
# coef[0] será o declive (slope) e coef[1] será a intercepção (intercept) da linha y = mx + b.
coef = np.polyfit(x, y, 1)

# Criar um objeto de função polinomial a partir dos coeficientes calculados.
# Isto permite calcular facilmente os valores y da linha de regressão para quaisquer valores x.
linha_regressao = np.poly1d(coef)

# Gerar valores de x uniformemente espaçados entre o mínimo e o máximo de 'Industrial Water Use (%)'.
# Estes valores serão usados para desenhar a linha de regressão de forma suave.
x_vals = np.linspace(x.min(), x.max(), 100)

# Plotar a linha de regressão usando os x_vals gerados e a função linha_regressao(x_vals).
# color='red': Define a cor da linha.
# label=...: Cria um rótulo para a legenda que inclui a equação da linha formatada.
plt.plot(x_vals, linha_regressao(x_vals), color='red', linewidth=2,
         label=f'Regressão Linear: y = {coef[0]:.3f}x + {coef[1]:.3f}') # Mais casas decimais para precisão

# Adicionar rótulos aos eixos e título ao gráfico.
plt.xlabel("Utilização Industrial de Água (%)")
plt.ylabel("Taxa de Esgotamento das Águas Subterrâneas (%)")
plt.title("Relação entre Uso Industrial de Água e Taxa de Esgotamento das Águas Subterrâneas")

# Adicionar legenda e grelha.
plt.legend()
plt.grid(True)
plt.tight_layout()

# Exibir o gráfico.
print(f"--- Tarefa 5 Concluída ---")
print("A exibir o gráfico de dispersão com regressão linear...")
plt.show()

# --- Interpretação da Regressão Linear ---
print("Interpretação da Regressão Linear:")
# O coeficiente coef[0] (declive) indica a mudança média esperada em Y para uma mudança unitária em X.
print(f"- O declive da linha de regressão é aproximadamente {coef[0]:.3f}.")
if coef[0] > 0.01:
    print("  Isto sugere que, em média, um aumento de 1 ponto percentual na Utilização Industrial de Água está associado a um aumento de aproximadamente "
          f"{coef[0]:.3f} pontos percentuais na Taxa de Esgotamento das Águas Subterrâneas, considerando os dados disponíveis.")
elif coef[0] < -0.01:
    print("  Isto sugere que, em média, um aumento de 1 ponto percentual na Utilização Industrial de Água está associado a uma diminuição de aproximadamente "
          f"{-coef[0]:.3f} pontos percentuais na Taxa de Esgotamento das Águas Subterrâneas, considerando os dados disponíveis.")
else:
    print("  Isto sugere que há pouca ou nenhuma associação linear entre a Utilização Industrial de Água e a Taxa de Esgotamento das Águas Subterrâneas "
          "nestes dados.")
# O coeficiente coef[1] (intercepção) é o valor esperado de Y quando X é 0.
print(f"- A intercepção é aproximadamente {coef[1]:.3f}.")
print("  Este valor representa a Taxa de Esgotamento das Águas Subterrâneas esperada quando a Utilização Industrial de Água é 0%, "
      "embora esta interpretação possa não ser prática ou significativa dependendo do contexto dos dados.")
print("- É importante notar que correlação não implica causalidade. Esta análise apenas mostra uma associação linear nos dados.")
print("----------------------------------------------------")


# --- Tarefa 6: Previsão com Machine Learning (Regressão Linear) ---
print(f"--- Tarefa 6: Machine Learning para Prever 'Per Capita Water Use (Liters per Day)' ---")

# --- Preparação dos Dados ---
# Selecionar as colunas que serão usadas como 'features' (variáveis preditoras) e a coluna 'target' (variável a prever).
features = ['Country', 'Agricultural Water Use (%)', 'Rainfall Impact (Annual Precipitation in mm)']
target = 'Per Capita Water Use (Liters per Day)'

# Criar um novo DataFrame contendo apenas as colunas de interesse.
# .dropna() remove linhas que tenham valor nulo em *qualquer* uma destas colunas,
# pois modelos de ML geralmente não lidam bem com dados em falta.
df_ml = df[features + [target]].dropna()
print(f"Número de amostras após remover nulos nas colunas de interesse: {len(df_ml)}")

# --- Engenharia de Features: One-Hot Encoding ---
# A variável 'Country' é categórica (texto). Modelos de regressão linear exigem dados numéricos.
# Usamos One-Hot Encoding para converter a coluna 'Country' em múltiplas colunas binárias (0 ou 1).
# Cada nova coluna representa um país específico.
# pd.get_dummies faz esta conversão automaticamente.
# drop_first=True remove a primeira categoria para evitar multicolinearidade (uma coluna pode ser inferida das outras).
df_ml_encoded = pd.get_dummies(df_ml, columns=['Country'], drop_first=True, prefix='Country')
print(f"Colunas após One-Hot Encoding da variável 'Country': {df_ml_encoded.columns.tolist()}")

# --- Divisão dos Dados: Treino e Teste ---
# Separar o DataFrame codificado em 'X' (features) e 'y' (target).
# X contém todas as colunas exceto a 'target'.
# y contém apenas a coluna 'target'.
X = df_ml_encoded.drop(target, axis=1)
y = df_ml_encoded[target]

# Dividir os dados X e y em conjuntos de treino e teste.
# O conjunto de treino é usado para 'ensinar' o modelo.
# O conjunto de teste é usado para avaliar o desempenho do modelo em dados não vistos.
# test_size=0.2 significa que 20% dos dados serão usados para teste, e 80% para treino.
# random_state=42 garante que a divisão seja a mesma sempre que o código for executado (reprodutibilidade).
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

# --- Treino do Modelo: Regressão Linear ---
# Técnica utilizada: Regressão Linear Múltipla.
# Este modelo tenta encontrar a melhor relação linear entre as features (X) e o target (y).
# Assume-se que y pode ser previsto como uma soma ponderada das features mais uma constante (intercepção).
# y = b0 + b1*X1 + b2*X2 + ... + bn*Xn

# Criar uma instância do modelo de Regressão Linear.
modelo = LinearRegression()

# Treinar o modelo usando os dados de treino (X_train, y_train).
# O método .fit() ajusta os coeficientes (pesos) do modelo para minimizar a diferença
# entre as previsões do modelo e os valores reais de y_train.
print("A treinar o modelo de Regressão Linear...")
modelo.fit(X_train, y_train)
print("Modelo treinado com sucesso.")

# Opcional: Exibir os coeficientes aprendidos pelo modelo
print("Coeficientes do modelo (pesos das features):")
# print(pd.DataFrame(modelo.coef_, index=X.columns, columns=['Coeficiente']))
# print(f"Intercepção do modelo: {modelo.intercept_:.2f}")

# --- Avaliação do Modelo ---
# Fazer previsões no conjunto de teste (dados que o modelo não viu durante o treino).
y_pred = modelo.predict(X_test)

# Calcular métricas de avaliação para comparar os valores previstos (y_pred) com os valores reais (y_test).
# - Mean Squared Error (MSE): Média dos quadrados das diferenças entre previsão e real. Penaliza erros grandes.
#   Quanto menor, melhor. A unidade é o quadrado da unidade do target.
mse = mean_squared_error(y_test, y_pred)
# - R-squared (R²): Coeficiente de determinação. Indica a proporção da variância no target que é
#   explicável pelas features no modelo. Varia entre -inf e 1.
#   Um valor de 1 significa previsão perfeita. Um valor de 0 significa que o modelo não é melhor
#   que simplesmente prever a média do target. Valores negativos indicam um modelo muito mau.
r2 = r2_score(y_test, y_pred)

print("--- Avaliação do Desempenho do Modelo no Conjunto de Teste ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.3f}")

# Interpretação dos Resultados:
print("Interpretação dos Resultados da Previsão:")
print(f"- O MSE de {mse:.2f} indica o erro quadrático médio das previsões.")
print(f"- O R² de {r2:.3f} sugere que aproximadamente {r2*100:.1f}% da variabilidade no '{target}' "
      f"pode ser explicada pelas variáveis 'Country', 'Agricultural Water Use (%)', e 'Rainfall Impact (Annual Precipitation in mm)' "
      f"através deste modelo linear, nos dados de teste.")
if r2 > 0.7:
    print("  Este é um valor de R² relativamente alto, indicando um bom ajuste do modelo aos dados.")
elif r2 > 0.4:
    print("  Este valor de R² indica um ajuste moderado do modelo aos dados.")
else:
    print("  Este valor de R² indica um ajuste fraco do modelo. Outras variáveis ou modelos mais complexos podem ser necessários para uma melhor previsão.")

print("--- Tarefa 6 Concluída ---")
print("----------------------------------------------------")
