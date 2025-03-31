"""

Bernardo Almeida, al78403
David Fidalgo, al79881
Tiago Valente al78800
Vasco Macedo al78798

Trabalho Experimental 1, "Global Water Consumption Dataset (2000-2024)":
  1. Carrega o ficheiro CSV completo, exibe informações iniciais e cria um novo DataFrame
     filtrado por países específicos (Italy, Japan, Spain, UK, USA), guardando-o num novo CSV.
  2. Plota a evolução do "Total Water Consumption" ao longo dos anos para os países selecionados.
  3. Cria um gráfico circular (pie chart) mostrando a distribuição percentual dos usos de água
     (agrícola, industrial e doméstico) para Spain no ano de 2020.
  4. Define uma função que retorna o ano e o valor da menor percentagem de "Agricultural Water Use"
     para um país fornecido como entrada.
  5. Plota um gráfico de dispersão com uma linha de regressão linear entre "Industrial Water Use" e
     "Groundwater Depletion Rate", utilizando o dataset completo.
  6. Utiliza técnicas de Machine Learning (Regressão Linear) para prever "Per Capita Water Use"
     com base nas variáveis "Country", "Agricultural Water Use" e "Rainfall Impact".

"""
import pandas as pd  
import matplotlib.pyplot as matplt  
import numpy as np  
from sklearn.model_selection import train_test_split  # Para dividir dados em treino e teste (ML)
from sklearn.metrics import mean_squared_error, r2_score  # Para avaliar o modelo de ML
from sklearn.linear_model import LinearRegression  # Para criar o modelo de Regressão Linear (ML)
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# --- Tarefa 1: Carregar, Filtrar e Guardar Dados ---
df = pd.read_csv("cleaned_global_water_consumption.csv")


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
matplt.figure(figsize=(12, 7))
# Iterar sobre cada país na lista 'paises'.
for pais in paises:
    # Filtrar o DataFrame 'df_filtrado' para obter os dados apenas do país atual.
    df_pais = df_filtrado[df_filtrado['Country'] == pais]
    # Plotar a evolução: 'Year' no eixo X, 'Total Water Consumption' no eixo Y.
    # marker='o' adiciona um marcador circular em cada ponto de dados.
    # label=pais define o nome que aparecerá na legenda para esta linha.
    matplt.plot(df_pais['Year'], df_pais['Total Water Consumption (Billion Cubic Meters)'], marker='o', linestyle='-', label=pais)

# Adicionar rótulos aos eixos X e Y e um título ao gráfico.
matplt.xlabel("Ano")
matplt.ylabel("Consumo Total de Água (Bilhões de Metros Cúbicos)")
matplt.title("Evolução Anual do Consumo Total de Água (2000-2024) por País Selecionado")
matplt.legend()
matplt.grid(True)

matplt.tight_layout()

print(f"--- Tarefa 2 Concluída ---")
print("A exibir o gráfico da evolução do consumo total de água...")
matplt.show()
print("----------------------------------------------------")


# --- Tarefa 3: Gráfico Circular (Pie Chart) para Spain em 2020 ---

# Utiliza-se um bloco try-except para lidar com o caso de não existirem dados para Spain em 2020.
try:
    dados_spain_2020 = df[(df['Country'] == 'Spain') & (df['Year'] == 2020)].iloc[0]
    labels = ['Uso Agrícola (%)', 'Uso Industrial (%)', 'Uso Doméstico (%)']
    valores = [
        dados_spain_2020['Agricultural Water Use (%)'],
        dados_spain_2020['Industrial Water Use (%)'],
        dados_spain_2020['Household Water Use (%)']
    ]

    matplt.figure(figsize=(7, 7))
    matplt.pie(valores, labels=labels, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'}) 
    matplt.title("Distribuição Percentual do Uso de Água em Espanha (2020)")

    print(f"--- Tarefa 3 Concluída ---")
    print("A exibir o gráfico circular da distribuição do uso de água em Espanha (2020)...")
    matplt.show()
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
    
    df_country = df_completo[df_completo['Country'] == nome_pais]
    if df_country.empty:
        print(f"Aviso: Não foram encontrados dados para o país '{nome_pais}'.")
        return None, None 
    
    idx_min = df_country['Agricultural Water Use (%)'].idxmin()

    linha_min = df_country.loc[idx_min]

    ano_min = linha_min['Year']
    valor_min = linha_min['Agricultural Water Use (%)']

    return int(ano_min), valor_min 

print(f"--- Tarefa 4: Teste da Função ---")
pais_selecionado = input("Digite um país:")

ano, valor = menor_uso_agricola(df, pais_selecionado)

if ano is not None:
    print(f"Para o país '{pais_selecionado}', o menor valor da coluna do uso da água na agricultura foi ({valor:.2f}%) ocorreu no ano {ano}.")
else:
    print(f"País selecionado não encontrado")
print("----------------------------------------------------")


# --- Tarefa 5: Gráfico de Dispersão e Regressão Linear ---

# Selecionar as colunas relevantes ('Industrial Water Use (%)', 'Groundwater Depletion Rate (%)') do DataFrame completo.
# .dropna() remove as linhas onde qualquer um destes valores seja nulo, para evitar erros na regressão.
df_scatter = df[['Industrial Water Use (%)', 'Groundwater Depletion Rate (%)']].dropna()

# Criar a figura e os eixos para o gráfico de dispersão.
matplt.figure(figsize=(10, 6))

# Criar o gráfico de dispersão:
# - Eixo X: 'Industrial Water Use (%)'.
# - Eixo Y: 'Groundwater Depletion Rate (%)'.
# - alpha=0.6: Define a transparência dos pontos (útil se houver sobreposição).
# - label='Dados': Rótulo para a legenda.
matplt.scatter(df_scatter['Industrial Water Use (%)'], df_scatter['Groundwater Depletion Rate (%)'], alpha=0.6, label='Dados Observados')

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
matplt.plot(x_vals, linha_regressao(x_vals), color='red', linewidth=2,
         label=f'Regressão Linear: y = {coef[0]:.3f}x + {coef[1]:.3f}') # Mais casas decimais para precisão

# Adicionar rótulos aos eixos e título ao gráfico.
matplt.xlabel("Utilização Industrial de Água (%)")
matplt.ylabel("Taxa de Esgotamento das Águas Subterrâneas (%)")
matplt.title("Relação entre Uso Industrial de Água e Taxa de Esgotamento das Águas Subterrâneas")

# Adicionar legenda e grelha.
matplt.legend()
matplt.grid(True)
matplt.tight_layout()

# Exibir o gráfico.
print(f"--- Tarefa 5 Concluída ---")
print("A exibir o gráfico de dispersão com regressão linear...")
matplt.show()

print("----------------------------------------------------")


# --- Tarefa 6: Previsão com Machine Learning (Regressão Linear) ---
features = ['Country', 'Agricultural Water Use (%)', 'Rainfall Impact (Annual Precipitation in mm)']
target = 'Per Capita Water Use (Liters per Day)'
df_ml = df[features + [target]].dropna()

# 2. Converter a variável categórica "Country" para numérica via One-Hot Encoding.
df_ml_encoded = pd.get_dummies(df_ml, columns=['Country'], drop_first=True)

# 3. Dividir os dados em conjuntos de treino (80%) e teste (20%).
X = df_ml_encoded.drop(target, axis=1)
y = df_ml_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Treinar o modelo escolhido. Aqui encontram-se 3 diferentes: Regressão Linear, Lasso e ElasticNet.
#model = LinearRegression()
#model = Lasso(alpha=1.0)
model = ElasticNet(alpha=1.0, l1_ratio=0.50)
model.fit(X_train, y_train)

# 5. Avaliar o desempenho do modelo.
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Exibir resultados e interpretações.
print("Previsão de 'Per Capita Water Use'")
print(f"Número total de amostras: {len(df_ml)}")
print(f"Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R²): {r2:.3f}")


print("--- Tarefa 6 Concluída ---")
print("----------------------------------------------------")
