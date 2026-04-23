# -*- coding: utf-8 -*-
"""
Trabalho de Conclusão de Curso apresentado para obtenção do título de especialista em 
Data Science e Analytics – 2026 - MBA-USP/Esalq
Fred Ricardo de Souza Costa 
"""
# %% Importar as bibliotecas

import pandas as pd
import seaborn as sns
import numpy as np
import geopandas as gpd
#import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = 'browser'
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from scipy.stats import zscore
from sklearn.cluster import KMeans
import plotly.io as pio
pio.renderers.default='browser'

# %% Ingestao a base de dados
# Base SINAN - Agravos de Notificação
print("Ingestao da base do SINAN Dengue 2025...")
data_types = {
    'CD_MUNICIP': 'Int32',
    'SG_UF_NOT': 'string',
    'ID_REGIONA': 'string',
    'ID_MUNICIP': 'Int32',
    'COUFINF': 'string',
    'COMUNINF': 'Int32',
    'ID_UNIDADE': 'string'
}
ds_dengue25 = pd.read_csv("../data/raw/DENGBR25.csv", sep=",", decimal=".", 
                          dtype=data_types, 
                          parse_dates=['DT_NOTIFIC', 'DT_SIN_PRI'])

# Dados estatisticos CVE distribuição dos casos autóctones de dengue segundo o
# município de residência, por semana epidemiológica de início de sintomas
print("Ingestao da base Dados estatisticos CVE...")
ds_dengue25_se_vigilancia = pd.read_csv("../data/raw/dengue25_se_vigilancia.csv",
                                        sep=";", decimal=".",
                                        dtype={'CD_MUNICIPIO_SINAN': 'Int32',
                                               'GVE': 'Int32',
                                               'DRS': 'Int32',
                                               'REGIAO_SAUDE': 'Int32'})

# Populacao estimada para 2025
# Arquivo csv tratado direto no arquivo, nome das colunas e concatena o codigo do municipio do ibge
print("Ingestao da base Populacao estimada para 2025...")
ds_pop_ibge = pd.read_csv("../data/raw/pop_munic_ibge.csv", sep=";", thousands=".", 
                          dtype={'CD_MUNICIPIO_IBGE':'string'})
ds_pop_ibge.drop(columns=['SG_UF','CD_UF_IBGE','COD_MUNIC','NM_MUNICIPIO_IBGE'], inplace=True)


# Códigos oficiais de Populacao TSE e IBGE
print("Ingestao da base Códigos oficiais de Populacao estimada para 2025...")
ds_tse_ibge = pd.read_csv("../data/raw/municipio_tse_ibge.csv", 
                          sep=";", decimal=".", encoding='latin1', 
                          dtype={'CD_MUNICIPIO_IBGE':str})


# Calendário Epidemiológico 2025
print("Ingestao da base Calendário Epidemiológico 2025...")
ds_cal_ipem = pd.read_csv("../data/raw/calendario_epidemiologico_2025.csv", 
                          sep=";", decimal=".", encoding='latin1')


# Importando mapa de sao paulo
print("Importando Mapa de SP...")
mapa_ibge_sp = "zip://../data/raw/SP_Municipios_2025.zip"

# mapa geométrico
mapa_sp = gpd.read_file(mapa_ibge_sp)

# %% Data Wrangling  

# Compatibilizando o tipo do codigo do municipio
mapa_sp['CD_MUN'] = mapa_sp['CD_MUN'].fillna(0).astype('Int32')

# Otimizando tipos
col_category = ['SG_UF_NOT', 'COUFINF', 'ID_REGIONA', 'ID_UNIDADE']
ds_dengue25[col_category] = ds_dengue25[col_category].astype('category')

# Ajsute das colunas que pelo dicionario sao do tipo char(1) = 1 – Sim 2 – Não 9 – Ignorado
# mapeando para Binario ? 1 (Sim) -> 1, 2 (Não) -> 0, 9 (Ignorado) -> Nulo (NaN)
# Tratamento efetuado so para a base de SP mas poder[a ser feito na base principal]
mapeamento_binario = {
    1: 1, 
    2: 0, 
    9: np.nan
}
col_SN = [
    'FEBRE', 'MIALGIA', 'CEFALEIA', 'EXANTEMA', 'VOMITO', 
    'NAUSEA', 'DOR_COSTAS', 'CONJUNTVIT', 'ARTRITE', 'ARTRALGIA',
    'PETEQUIA_N', 'LEUCOPENIA', 'LACO', 'DOR_RETRO', 'HOSPITALIZ'
]

for coluna in col_SN:
    if coluna in ds_dengue25.columns:
        # O map faz a substituição e o tipo Int8 garante o armazenamento leve de 1, 0 e nulos
        ds_dengue25[coluna] = ds_dengue25[coluna].map(mapeamento_binario).astype('Int8')


# O SINAN desconsidera o digito verificador
ds_tse_ibge['CD_MUNICIPIO_SINAN']= ds_tse_ibge['CD_MUNICIPIO_IBGE'].str[:-1]

#Agregando a poupulacao estimada para 2025
ds_tse_ibge = pd.merge(
    ds_tse_ibge,
    ds_pop_ibge,
    on='CD_MUNICIPIO_IBGE',
    how='left')

# Otimizando os tipos
ds_tse_ibge['CD_MUNICIPIO_SINAN'] = pd.to_numeric(ds_tse_ibge['CD_MUNICIPIO_SINAN'],
                                                 errors='coerce').fillna(0).astype('Int32')
ds_tse_ibge['CD_MUNICIPIO_IBGE'] = pd.to_numeric(ds_tse_ibge['CD_MUNICIPIO_IBGE'], 
                                                 errors='coerce').fillna(0).astype('Int32')

# %% Transformacao dos dados

# Separacao do dataset para Sao Paulo codigo IBGE 35
df_tse_ibge_sp  = ds_tse_ibge.query('CD_UF_IBGE ==35')
df_tse_ibge_sp.info()

# divisao das principais unidades orgnaizacionais de saude
# DRS - Departamentos Regionais de Saúde
# GVE - Grupos de Vigilância Epidemiológica
# Regiões de Saúde
df_unidade_org_saude = pd.merge(
    ds_dengue25_se_vigilancia [['DRS','NOME_DRS','GVE','NOME_GVE',
                               'REGIAO_SAUDE','NOME_REIGIAO_SAUDE','CD_MUNICIPIO_SINAN']],
    df_tse_ibge_sp[['CD_MUNICIPIO_SINAN','CD_MUNICIPIO_IBGE']],
    left_on='CD_MUNICIPIO_SINAN',
    right_on='CD_MUNICIPIO_SINAN',
    how='left'
)

df_unidade_org_saude.info()

"""
Para garantir a integridade da análise de difusão espacial intraestadual, 
aplicou-se um filtro de procedência da infecção. Foram excluídos os casos
importados (pacientes notificados em SP, mas cuja UF de infecção - 
COUFINF - apontava para outros estados). 
"""

# Fazer o tratamento da cidade real de notificacao ou possivel considerando que
# se a cidade de infeccao possivel nao for informada considerar a cidade da notificacao
mun_infeccao = pd.to_numeric(ds_dengue25['COMUNINF'], errors='coerce').astype(str)
mun_notificacao = pd.to_numeric(ds_dengue25['ID_MUNICIP'],errors='coerce').astype(str)

vazios_sinan = [' ', '', '000000', '00000', 'nan','0']
mun_infeccao = ds_dengue25['COMUNINF'].astype(str).replace(vazios_sinan, np.nan)
mun_notificacao = ds_dengue25['ID_MUNICIP'].astype(str).replace(vazios_sinan, np.nan)

ds_dengue25['CD_MUNICIPIO_INFEC'] = mun_infeccao.fillna(mun_notificacao)

"""
Foram retidos na amostra apenas os casos com COUFINF='35' ou aqueles notificados 
em SP com campo COUFINF ignorado/nulo, assumindo-se, nestes últimos, a infecção autóctone.
"""
# classificações finais de casos confirmados de dengue 
# 10 para Dengue Clássica
# 11 para Dengue com Sinais de Alarme
# 12 para Dengue Grave

# Para considerar o dados que envolvem as cidades com possivel infeccao
# df_dengue25_sp = ds_dengue25[
#     (ds_dengue25['CLASSI_FIN'] >= 10) & 
#     (ds_dengue25['CD_MUNICIPIO_INFEC'].str.startswith('35', na=False))
# ].copy()

# Considerando somente as notificacoes de SP
df_dengue25_sp = ds_dengue25[
    (ds_dengue25['CLASSI_FIN'] >= 10) & 
    (ds_dengue25['SG_UF_NOT'].str.startswith('35', na=False))
].copy()


df_dengue25_sp['CD_MUNICIPIO_INFEC'] = pd.to_numeric(df_dengue25_sp['CD_MUNICIPIO_INFEC'], 
                                                 errors='coerce').fillna(0).astype('Int32')

# Retirando colunas vazias
df_dengue25_sp.dropna(axis=1, how='all', inplace=True)

# trazendo o codigo municipio completo do IBGE e Divisoes de saude
df_dengue25_sp = pd.merge(
    df_dengue25_sp,
    df_unidade_org_saude[['CD_MUNICIPIO_SINAN',
                          'CD_MUNICIPIO_IBGE',
                          'DRS', 
                          'GVE', 
                          'REGIAO_SAUDE']],
    left_on='CD_MUNICIPIO_INFEC',
    right_on='CD_MUNICIPIO_SINAN',
    how='left'
)

df_dengue25_sp = df_dengue25_sp.dropna(subset=['GVE', 
                                               'DRS', 
                                               'REGIAO_SAUDE',
                                               'CD_MUNICIPIO_IBGE']).copy()

df_dengue25_sp = df_dengue25_sp.drop(columns=['CD_MUNICIPIO_SINAN'])


df_descr_dengue25_sp = df_dengue25_sp.describe().T
ds_dengue25.info()
df_dengue25_sp.info()

# %% Paleta de cores
paleta_k4 = {
    0: '#90CAF9',  # Azul Gelo
    1: '#F48FB1',  # Rosa Queimado
    2: '#A5D6A7',  # Verde Menta
    3: '#CE93D8',  # Lavanda
}


paleta_k6 = {
    0: '#90CAF9',  # Azul Gelo
    1: '#F48FB1',  # Rosa Queimado
    2: '#A5D6A7',  # Verde Menta
    3: '#CE93D8',  # Lavanda
    4: '#80DEEA',   # Turquesa Suave
    5: '#FFCC80'  # Damasco
}

# 1. Garante que as cores sejam passadas para o mapa na ordem correta (0, 1, 2, 3)
cores_ord_k4 = [paleta_k4[i] for i in sorted(paleta_k4.keys())]
cores_ord_k6 = [paleta_k6[i] for i in sorted(paleta_k6.keys())]


# 2. Cria o mapa de cores customizado
k4_cmap = ListedColormap(cores_ord_k4)
k6_cmap = ListedColormap(cores_ord_k6)
# %% Explorar as quantidades de notificacao em funcao da semana epimideologica (SE)

# Analise da semana epidemiologia
df_notif_SE = df_dengue25_sp[['DT_NOTIFIC', 
                              'SEM_NOT', 
                              'SG_UF_NOT', 
                              'CD_MUNICIPIO_INFEC', 
                              'CD_MUNICIPIO_IBGE',
                              'GVE',
                              'ID_REGIONA', 
                              'DT_SIN_PRI', 
                              'SEM_PRI']]
df_notif_SE.info()
df_notif_SE['SEM_NOT'] = df_notif_SE['SEM_NOT'].astype(str).str[4:].astype(int)
df_notif_SE.sort_values(by='SEM_NOT', inplace = True, ascending=True)
df_notif_SE['SEM_PRI'] = df_notif_SE['SEM_PRI'].astype(str).str[4:].astype(int)
df_notif_SE.sort_values(by='SEM_PRI', inplace = True, ascending=True)

# Formato Padrao TCC
fig, ax = plt.subplots(figsize=(14, 6))

# Garantir que não há preenchimento (fundo branco limpo)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plotagem dos dados
ax = sns.countplot(data=df_notif_SE, x="SEM_PRI", palette='viridis')

# Formatação dos Títulos dos Eixos (Arial, tam 11, preto)
plt.xlabel('Semana Epidemiológica (Primeiros Sintomas)', 
           fontsize=11, color='black', fontfamily='Arial', labelpad=10)
plt.ylabel('Notificações', 
           fontsize=11, color='black', fontfamily='Arial', labelpad=10)

# Ticks
plt.xticks(rotation=60, ha='right', fontsize=10, fontfamily='Arial')
plt.yticks(fontsize=10, fontfamily='Arial')

# Linhas de grade (Não deve conter)
ax.grid(False)

# Remoção de bordas e formatação dos Eixos (Sólida, Preta, 1.5pt)
ax.spines['top'].set_visible(False)    # Remove borda superior
ax.spines['right'].set_visible(False)  # Remove borda direita
ax.spines['bottom'].set_linewidth(1.5) # Eixo X
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1.5)   # Eixo Y
ax.spines['left'].set_color('black')

plt.show()

# %% Efetuando analise time series da base de notificacoes absoluta
df_notif_SE_dia = df_notif_SE.groupby(by=['SEM_PRI','DT_SIN_PRI']).size().reset_index(name='TOTAL_NOTIFIC_DIA')
# analise de Tendencia por media movel 7 dias 
df_notif_SE_dia['MEDIA_MOV']= df_notif_SE_dia['TOTAL_NOTIFIC_DIA'].rolling(window=7).mean()

ts_dengue25_sp = df_notif_SE_dia.set_index('DT_SIN_PRI')['TOTAL_NOTIFIC_DIA']

# ===== Formato TCC
fig, ax = plt.subplots(figsize=(12, 6))

# Remoção de preenchimento (Fundo branco)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plotagens
ax.plot(df_notif_SE_dia['DT_SIN_PRI'], df_notif_SE_dia['TOTAL_NOTIFIC_DIA'], label='Por Dia')
ax.plot(df_notif_SE_dia['DT_SIN_PRI'], df_notif_SE_dia['MEDIA_MOV'], label='Media Movel (7 dias)', color='red')

# Títulos dos Eixos (Sem título principal, fonte Arial 11, preta)
ax.set_xlabel('Data dos Primeiros Sintomas', fontsize=11, color='black', fontfamily='Arial', labelpad=10)
ax.set_ylabel('Comportamento da Dengue 2025', fontsize=11, color='black', fontfamily='Arial', labelpad=10)

# Legenda e Ticks em Arial
ax.legend(prop={'family': 'Arial', 'size': 10})
plt.xticks(rotation=90, fontfamily='Arial', fontsize=10)
plt.yticks(fontfamily='Arial', fontsize=10)

# Formatação de Bordas e Linhas de Eixo (1.5pt sólidas)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('black')

plt.show()

# %% Criando um mapa das unidades de saude 

# Unir a informação da Regional ao desenho dos municípios
mapa_unidades_saude = mapa_sp.merge(
    df_unidade_org_saude.drop_duplicates(),
    left_on='CD_MUN',
    right_on='CD_MUNICIPIO_IBGE',
    how='left'
)

def plot_mapa_unidade_saude(mapa, col_codigo, col_nome, titulo):
    mapa_plot = mapa.copy()
    
    col_legenda = f'LEGENDA_{col_codigo}'
    print (f'Construindo mapa {titulo}')
    mapa_plot[col_legenda] = mapa_plot[col_codigo].astype(str) + " - " + mapa_plot[col_nome].astype(str)
    
    # (aggfunc='first' para manter o texto da legenda após o dissolve)
    mapa_unidades = mapa_plot.dissolve(by=col_codigo, aggfunc='first')
    mapa_unidades = mapa_unidades[mapa_unidades.index.notna()]

    fig, ax = plt.subplots(figsize=(16, 12))

    # Plotagem com legenda horizontal e externa
    # ncol=4 distribui em 4 colunas 
    # loc='upper center' e bbox_to_anchor colocam a legenda abaixo do mapa
    mapa_unidades.plot(
        ax=ax, 
        column=col_legenda,
        categorical=True,
        legend=True,
        cmap='tab20', # tab20 é excelente para muitas categorias
        edgecolor='white', 
        linewidth=0.8,
        legend_kwds={
            'loc': 'upper center',
            'bbox_to_anchor': (0.5, -0.05), 
            'ncol': 4,                     
            'title': f"Divisão: {col_codigo}",
            'fontsize': 9,
            'frameon': False
        }
    )

    # Códigos nos centróides
    for idx, row in mapa_unidades.iterrows():
        centroid = row.geometry.centroid
        if not centroid.is_empty:
            ax.annotate(
                text=str(idx), 
                xy=(centroid.x, centroid.y), 
                horizontalalignment='center', 
                fontsize=8, 
                fontweight='bold',
                color='black'
            )

    ax.set_title(f"Mapa de Divisoes de Unidades de Saúde: {titulo}", fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.subplots_adjust(bottom=0.2) 
    plt.show()
    

# 1. Para DRS (Cód: DRS, Nome: NOME_DRS)
plot_mapa_unidade_saude(mapa_unidades_saude, 'DRS', 'NOME_DRS','DRS - Departamentos Regionais de Saúde')

# 2. Para GVE (Cód: GVE, Nome: NOME_GVE)
plot_mapa_unidade_saude(mapa_unidades_saude, 'GVE', 'NOME_GVE','GVE - Grupos de Vigilância Epidemiológica')

# 3. Para Região de Saúde (Cód: REGIAO_SAUDE, Nome: NOME_REIGIAO_SAUDE)
plot_mapa_unidade_saude(mapa_unidades_saude, 'REGIAO_SAUDE', 'NOME_REIGIAO_SAUDE','Regiões de Saúde')

# %% Analise por GVE - Grupos de Vigilância Epidemiológica
# Notificacao totalizada por GVE

df_notif_SE_gve = df_notif_SE.groupby(['GVE', 'SEM_PRI']).size().reset_index(name='TOTAL_NOTIFIC')
df_notif_SE_gve_media = df_notif_SE_gve.groupby('SEM_PRI')['TOTAL_NOTIFIC'].mean().reset_index(name='MEDIA_NOTIFIC')


# ====== Formato TCC
fig, ax = plt.subplots(figsize=(14, 6))

# Remoção de preenchimento (Fundo totalmente branco)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plotagem das linhas dos GVEs
ax = sns.lineplot(
    data=df_notif_SE_gve, 
    x="SEM_PRI", 
    y="TOTAL_NOTIFIC", 
    hue="GVE", 
    palette='viridis',  
    alpha=0.6, 
    marker='o',
    legend=False,
    linewidth=2
)

# Plotagem da Média Estadual
ax.plot(
    df_notif_SE_gve_media['SEM_PRI'],
    df_notif_SE_gve_media['MEDIA_NOTIFIC'],
    color='red',
    linewidth=2,
    linestyle='--',
    marker='D',
    markersize=8,
    label='MÉDIA ESTADUAL'
)

# Etiqueta de texto no final da linha (Média Estadual)
ultima_semana = 53
altura_media = df_notif_SE_gve_media[df_notif_SE_gve_media['SEM_PRI'] == ultima_semana]['MEDIA_NOTIFIC'].values[0]

ax.text(
    x=ultima_semana + 0.5, 
    y=altura_media, 
    s="MÉDIA ESTADUAL", 
    fontsize=8, 
    fontweight='bold',
    color='red',
    va='center',
    fontfamily='Arial' # Padronizando a fonte do texto solto
)

# Formatação dos Títulos dos Eixos (Sem título do gráfico, Arial 11, Preta)
plt.xlabel('Semana Epidemiológica', fontsize=11, color='black', fontfamily='Arial', labelpad=10)
plt.ylabel('Notificações', fontsize=11, color='black', fontfamily='Arial', labelpad=10)

# 6. Formatação dos Ticks (Valores dos eixos) em Arial
plt.xticks(ticks=np.arange(1, 53, 1), rotation=60, ha='right', fontsize=10, fontfamily='Arial')
plt.yticks(fontsize=10, fontfamily='Arial')

# Regras rigorosas da Tabela 8 (Grades, Bordas e Eixos 1.5pt)
# Formatação de Bordas e Linhas de Eixo (1.5pt sólidas)
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1.5)
ax.spines['left'].set_color('black')


plt.tight_layout()
plt.show()

# %% Analise de agrupamento com Clustering Dynamic Time Warping (DTW)
# Preparando a Matriz por GVE
# Criando a matriz 

matriz_gve = df_notif_SE_gve.pivot(
    index='GVE', 
    columns='SEM_PRI', 
    values='TOTAL_NOTIFIC'
).fillna(0) # Zera as semanas sem casos

matriz_gve_cluster = matriz_gve.copy()

# %% Padronizacao com Scaler 
#  faz a mesma coisa que o zscore com ddof=0, que é o padrão de ML
scaler = TimeSeriesScalerMeanVariance()

# Z-score na nossa matriz de GVEs
# transforma os dados brutos (0 a milhares) em desvios padrão (ex: -1.5 a +3.0)
matriz_gve_pad = scaler.fit_transform(matriz_gve.values)

print("Dados padronizados! Treinando o modelo DTW...")


# %% Testando a quantidade de cluster utilizando Elbow Method DTW

print("Iniciando o Elbow Method com DTW")

# Lista para guardar os custos (Inércia DTW) de cada teste
inercia_dtw = []

# Teste divisão do Estado de São Paulo entre 2 e 8 perfis epidemiológicos
K_range = range(2, 9) 

for k in K_range:
    # O n_jobs=-1 é paralelizar o processamento
    modelo = TimeSeriesKMeans(n_clusters=k, metric="dtw", random_state=42, n_jobs=-1)

    modelo.fit(matriz_gve.values)
    
    # Extrai o custo total de alinhamento (Inércia) e guarda na lista
    inercia_dtw.append(modelo.inertia_)
    
    print(f"Teste com {k} Clusters concluído. Custo DTW: {modelo.inertia_:.0f}")

print("Testes finalizados! Desenhando o gráfico...")

# ===== Formato TCC
# Desenhando o Gráfico do Elbow
fig, ax = plt.subplots(figsize=(10, 6))

# Remoção de preenchimento (Fundo totalmente branco)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plotagem da linha do Elbow
ax.plot(K_range, inercia_dtw, marker='o', color='darkblue', linewidth=2, markersize=8)

# Formatação dos Títulos dos Eixos (Sem título do gráfico, Arial 11, Preta)
ax.set_xlabel("Número de Clusters (k)", fontsize=11, color='black', fontfamily='Arial', labelpad=10)
ax.set_ylabel("Soma do Custo DTW (Inércia)", fontsize=11, color='black', fontfamily='Arial', labelpad=10)

# Formatação dos Ticks (Valores dos eixos) em Arial
plt.xticks(fontfamily='Arial', fontsize=10)
plt.yticks(fontfamily='Arial', fontsize=10)

# Regras rigorosas da Tabela 8 (Grades, Bordas e Eixos 1.5pt)
ax.grid(False) # Não deve conter linhas de grade
ax.spines['top'].set_visible(False)    # Remove borda superior
ax.spines['right'].set_visible(False)  # Remove borda direita

ax.spines['bottom'].set_linewidth(1.5) # Eixo principal horizontal
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1.5)   # Eixo principal vertical
ax.spines['left'].set_color('black')

plt.tight_layout()
plt.show()

# %% Aplicando o modelo DTW 

n_k = 4
# Treinamos o modelo na matriz padronizada
modelo_dtw = TimeSeriesKMeans(n_clusters=n_k, metric="dtw", random_state=42, n_jobs=-1)
labels_clusters = modelo_dtw.fit_predict(matriz_gve_pad)

# Colamos o resultado de volta na nossa tabela
matriz_gve_cluster['CLUSTER_EPIDEMICO_DTW'] = labels_clusters
matriz_gve_cluster['CLUSTER_EPIDEMICO_DTW'] = matriz_gve_cluster['CLUSTER_EPIDEMICO_DTW'].astype('category')

# %% Mapa de Clusters DTW por GVE
mapa_unidades_saude['GVE'] = mapa_unidades_saude['GVE'].astype(str)
matriz_gve_cluster.index = matriz_gve.index.astype(str)

# Dissolve os municípios para criar os polígonos dos GVEs
mapa_dtw_gve = mapa_unidades_saude.dissolve(by='GVE')


# Junta o formato do mapa com o resultado DTW
mapa_dtw_gve = mapa_dtw_gve.merge(
    matriz_gve_cluster[['CLUSTER_EPIDEMICO_DTW']], 
    left_index=True, 
    right_index=True, 
    how='left'
)

#%% Plotagem do Mapa DTW
fig, ax = plt.subplots(figsize=(14, 10))
mapa_dtw_gve.plot(
    ax=ax,
    column='CLUSTER_EPIDEMICO_DTW',
    categorical=True,
    cmap=k4_cmap,          
    edgecolor='black',
    linewidth=0.5,
    legend=True,
)

mapa_dtw_gve['LEGENDA'] = (
                            mapa_dtw_gve['CLUSTER_EPIDEMICO_DTW'].astype(str).str.zfill(2) + 
                            " - " + mapa_dtw_gve.index.astype(str).str.zfill(2) +
                            " - " + mapa_dtw_gve['NOME_GVE'].astype(str)
                        )

legend_patches = []

# Loop para passar por cada um dos clusters
for i in range(n_k):
    dados_cluster = mapa_dtw_gve[mapa_dtw_gve['CLUSTER_EPIDEMICO_DTW'] == i]
    lista_legendas = dados_cluster['LEGENDA'].tolist()
    lista_legendas.sort()    
    # Só faz algo se o cluster não estiver vazio
    if len(lista_legendas) > 0:
        
        # A primeira linha recebe o quadradinho com a cor do cluster
        patch_principal = mpatches.Patch(color=paleta_k4[i], label=lista_legendas[0])
        legend_patches.append(patch_principal)
        
        # As linhas seguintes recebem um quadradinho "invisível" (color='none')
        # Isso mantém o recuo do texto alinhado perfeitamente!
        for texto_linha in lista_legendas[1:]:
            patch_invisivel = mpatches.Patch(color=paleta_k4[i], label=texto_linha)
            legend_patches.append(patch_invisivel)
            
        # Adiciona uma linha totalmente vazia no final do cluster para dar um respiro visual
        # (Mas não adiciona depois do último cluster para não sobrar espaço à toa)
        if i < 3: 
            espaco_vazio = mpatches.Patch(color='none', label='')
            legend_patches.append(espaco_vazio)

# Adiciona a legenda ao gráfico (A mesma configuração de antes)
ax.legend(
    handles=legend_patches, 
    title="Perfil da Epidemia (Cluster DTW)\n",
    loc='upper left', 
    bbox_to_anchor=(1.0, 1), # Mantém para fora do mapa
    prop={'family': 'Arial', 'size': 10}, 
    frameon=False, 
    labelspacing=0.3 # Deixa as linhas do mesmo cluster mais juntinhas
)
# Formata o título da legenda
plt.setp(ax.get_legend().get_title(), 
         fontproperties={'family': 'Arial', 'size': 10})


 # Códigos nos centróides
for idx, row in mapa_dtw_gve.iterrows():
     centroid = row.geometry.centroid
     if not centroid.is_empty:
         ax.annotate(
             text=str(idx), 
             xy=(centroid.x, centroid.y), 
             horizontalalignment='center', 
             fontsize=8, 
             color='black'
         )

ax.axis('off')
plt.tight_layout() # Garante que a legenda não seja cortada ao salvar a imagem
plt.show()
# %% Validacao DTW padronizada

print("Extraindo as médias morfológicas (DBA) do modelo DTW...")

# O modelo_dtw já guarda os DBA calculados dentro do atributo cluster_centers_
# Ele retorna no formato 3D (k, semanas, 1). O .squeeze() converte para (k, semanas)
centroides_dtw = modelo_dtw.cluster_centers_.squeeze()

# ====== Formato TCC
# Desenhando o painel com as 4 morfologias
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharey=True)
axes = axes.flatten()

# Garantir fundo branco total (Regra: sem preenchimento)
fig.patch.set_facecolor('white')

for i in range(n_k): # Assumindo k=4 clusters
    ax = axes[i]
    
    # Ajuste do fundo de cada painel
    ax.set_facecolor('white')
    
    # Todas as linhas (GVEs) que pertencem a este cluster
    gves_no_cluster = matriz_gve_pad[matriz_gve_cluster['CLUSTER_EPIDEMICO_DTW'] == i]

    # Desenha as curvas reais dos GVEs no fundo (clarinhas)
    for gve in gves_no_cluster:
        ax.plot(gve, color='gray', alpha=0.2, linewidth=1)
          
    # Desenha A MORFOLOGIA MÉDIA (O Centróide DBA) por cima
    ax.plot(centroides_dtw[i], color=paleta_k4[i], linewidth=3, label='Morfologia (DBA)')
    
    # Título do Painel: ALINHADO À ESQUERDA e em MAIÚSCULO
    titulo_painel = f'CLUSTER {i} (N={len(gves_no_cluster)} GVES)'
    ax.set_title(titulo_painel.upper(), loc='left', fontsize=12, fontweight='bold', color=paleta_k4[i], fontfamily='Arial')
    
    # Rótulos dos Eixos (Arial, Tam 11, Preto)
    ax.set_xlabel('Semana Epidemiológica', fontsize=11, color='black', fontfamily='Arial')
    ax.set_ylabel('Casos (Z-Score)', fontsize=11, color='black', fontfamily='Arial')
    
    # Linha zero (referência do Z-Score - mantida pois não é grade de fundo)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5) 
    
    # Formatação da Legenda (Arial)
    ax.legend(prop={'family': 'Arial', 'size': 10}, frameon=False)
    
    # Formatação dos Ticks (Arial, Tam 10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        
    # Regras da Tabela 8: Grades, Bordas e Eixos (1.5pt)
    ax.grid(False) # Não deve conter linhas de grade
    
    # Formatação das linhas dos eixos (Sólida, Preta, 1.5pt)
    ax.spines['top'].set_visible(False)    # Sem borda superior
    ax.spines['right'].set_visible(False)  # Sem borda direita
    
    ax.spines['bottom'].set_linewidth(1.5) # Eixo principal horizontal
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)   # Eixo principal vertical
    ax.spines['left'].set_color('black')

# Ajuste final de layout (Removido plt.suptitle para cumprir a Norma)
plt.tight_layout() 
plt.show()


# %% Validacao DTW com notificacoes

print("Extraindo as médias morfológicas (DBA) do modelo DTW...")

# ====== Formato TCC
# Desenhando o painel com as 4 morfologias (Volume Bruto - DTW)
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharey=True)
axes = axes.flatten()

# Garantir fundo branco total (Regra: sem preenchimento)
fig.patch.set_facecolor('white')

for i in range(n_k):
    ax = axes[i]
    
    # Ajuste do fundo de cada painel
    ax.set_facecolor('white')
    
    gves_no_cluster = (matriz_gve_cluster['CLUSTER_EPIDEMICO_DTW'] == i).values
    series_cluster_idx = matriz_gve[gves_no_cluster]
    
    # Desenha as linhas de todas as GVEs (com a cor da paleta)
    for s in series_cluster_idx.values:
        ax.plot(s, color=paleta_k4[i], linewidth=1) 
    
    # Título do Painel: ALINHADO À ESQUERDA e em MAIÚSCULO
    titulo_painel = f'CLUSTER {i} (N={len(series_cluster_idx)} GVES)'
    ax.set_title(titulo_painel.upper(), loc='left', fontsize=12, fontweight='bold', color=paleta_k4[i], fontfamily='Arial')
    
    # Rótulos dos Eixos (Arial, Tam 11, Preto)
    ax.set_xlabel('Semana Epidemiológica', fontsize=11, color='black', fontfamily='Arial')
    ax.set_ylabel('Casos Notificados', fontsize=11, color='black', fontfamily='Arial')
    
    # Linha zero (Referência)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.5) 
    
    # Formatação dos Ticks (Arial, Tam 10)
    ax.tick_params(axis='both', which='major', labelsize=10)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')

    # Regras da Tabela 8: Grades, Bordas e Eixos (1.5pt)
    ax.grid(False) # Não deve conter linhas de grade
    
    # Formatação das linhas dos eixos (Sólida, Preta, 1.5pt)
    ax.spines['top'].set_visible(False)    # Sem borda superior
    ax.spines['right'].set_visible(False)  # Sem borda direita
    
    ax.spines['bottom'].set_linewidth(1.5) # Eixo principal horizontal
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_linewidth(1.5)   # Eixo principal vertical
    ax.spines['left'].set_color('black')

# Ajuste final de layout (Removido plt.suptitle para cumprir a Norma)
plt.tight_layout() 
plt.show()


#%% ====== K-means 

matriz_gve_km = matriz_gve.copy()
# Padronização das variáveis cluster K-Means

# Aplicando o procedimento de ZScore em todas as variáveis
# As variáveis passam a ter média = 0 e desvio padrão = 1
matriz_gve_pad = matriz_gve_km.apply(zscore, ddof=1)

#%% Identificação da quantidade de clusters (Método Elbow) KMeans

elbow = []
K = range(1,11) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(matriz_gve)
    elbow.append(kmeanElbow.inertia_)
    
# ===== Formato TCC
fig, ax = plt.subplots(figsize=(16, 8))

# Remoção de preenchimento (Fundo totalmente branco)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Plotagem da linha (mantendo as variáveis K e elbow originais)
ax.plot(K, elbow, marker='o', color='darkblue', linewidth=2, markersize=8)

# Formatação dos Títulos dos Eixos (Sem título do gráfico, Arial 11, Preta)
ax.set_xlabel('Nº Clusters', fontsize=11, color='black', fontfamily='Arial', labelpad=10)
ax.set_ylabel('WCSS', fontsize=11, color='black', fontfamily='Arial', labelpad=10)

# Formatação dos Ticks (Valores dos eixos) em Arial
plt.xticks(ticks=range(1, 11), fontfamily='Arial', fontsize=10)
plt.yticks(fontfamily='Arial', fontsize=10)

# Regras rigorosas da Tabela 8 (Grades, Bordas e Eixos 1.5pt)
ax.grid(False) # Não deve conter linhas de grade
ax.spines['top'].set_visible(False)    # Remove borda superior
ax.spines['right'].set_visible(False)  # Remove borda direita
ax.spines['bottom'].set_linewidth(1.5) # Eixo principal horizontal
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1.5)   # Eixo principal vertical
ax.spines['left'].set_color('black')

plt.tight_layout()
plt.show()

#%% Execucao Cluster K-means

kmeans_final = KMeans(n_clusters = 4, init = 'random', random_state=100).fit(matriz_gve_pad)

# Gerando a variável para identificarmos os clusters gerados
kmeans_clusters = kmeans_final.labels_
matriz_gve_km['CLUSTER_EPIDEMICO_KM'] = kmeans_clusters
matriz_gve_cluster['CLUSTER_EPIDEMICO_KM'] = kmeans_clusters
matriz_gve_cluster['CLUSTER_EPIDEMICO_KM'] = matriz_gve_cluster['CLUSTER_EPIDEMICO_KM'].astype('category')
matriz_gve_km['CLUSTER_EPIDEMICO_KM'] = matriz_gve_km['CLUSTER_EPIDEMICO_KM'].astype('category')


# %% Validacao do KMEANS
# Obtendo os centróides matemáticos (curvas médias) do modelo KMeans
centroides_km = kmeans_final.cluster_centers_

# ==== Formato TCC
# Configuração da figura com 4 painéis
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
axes = axes.flatten()

# Garantir fundo branco total
fig.patch.set_facecolor('white')

for i in range(4):
    # Ajuste do fundo de cada painel
    axes[i].set_facecolor('white')
    
    # Filtro dos dados do cluster
    gves_no_cluster = (matriz_gve_cluster['CLUSTER_EPIDEMICO_KM'] == i).values
    series_cluster_idx = matriz_gve_pad[gves_no_cluster]
    
    # Desenho das linhas das GVEs (Cinza)
    for s in series_cluster_idx.values:
        axes[i].plot(s, color='gray', alpha=0.3, linewidth=1)
        
    # Desenho do Centróide (Média Aritmética)
    axes[i].plot(centroides_km[i].ravel(), color=paleta_k4[i], linewidth=3, label='Média Aritmética')
    
    # Título do Painel: ALINHADO À ESQUERDA e em MAIÚSCULO (Regra Adicional)
    titulo_painel = f'CLUSTER {i} (N={len(series_cluster_idx)} GVES)'
    axes[i].set_title(titulo_painel.upper(), loc='left', fontsize=12, fontweight='bold', color=paleta_k4[i], fontfamily='Arial')
    
    # Rótulos dos Eixos (Arial, Tam 11, Preto)
    axes[i].set_ylabel('CASOS PADRONIZADOS', fontsize=11, color='black', fontfamily='Arial')
    axes[i].set_xlabel('SEMANA EPIDEMIOLÓGICA', fontsize=11, color='black', fontfamily='Arial')
    
    # Formatação da Legenda (Arial)
    axes[i].legend(prop={'family': 'Arial', 'size': 10}, frameon=False)
    
    # === A CORREÇÃO ESTÁ AQUI ===
    # Linha zero (referência do Z-Score - referenciando axes[i])
    axes[i].axhline(0, color='black', linestyle='--', linewidth=0.5) 
    
    # Formatação dos Ticks (Arial, Tam 10)
    axes[i].tick_params(axis='both', which='major', labelsize=10)
    for label in (axes[i].get_xticklabels() + axes[i].get_yticklabels()):
        label.set_fontname('Arial')

    # Regras da Tabela 8: Grades, Bordas e Eixos (1.5pt)
    axes[i].grid(False) # Não deve conter linhas de grade
    
    # Formatação das linhas dos eixos (Sólida, Preta, 1.5pt)
    axes[i].spines['top'].set_visible(False)    # Sem borda superior
    axes[i].spines['right'].set_visible(False)  # Sem borda direita
    
    axes[i].spines['bottom'].set_linewidth(1.5) # Eixo X
    axes[i].spines['bottom'].set_color('black')
    axes[i].spines['left'].set_linewidth(1.5)   # Eixo Y
    axes[i].spines['left'].set_color('black')

# Ajuste final de layout (Removido plt.suptitle para cumprir a Norma de "Não conter título")
plt.tight_layout()
plt.show()
#%% Identificação das características dos clusters
# Agrupando o banco de dados
analise_gve_km = matriz_gve_km.groupby(by=['CLUSTER_EPIDEMICO_KM'])
# Estatísticas descritivas por grupo
tab_medias_grupo = analise_gve_km.mean().T
tab_desc_grupo = analise_gve_km.describe().T

# %% Mapa de Clusters Kmeans por GVE
# Dissolve os municípios para criar os polígonos dos GVEs
mapa_unidades_saude['GVE'] = mapa_unidades_saude['GVE'].astype(str)
matriz_gve_cluster.index = matriz_gve.index.astype(str)

# Dissolve os municípios para criar os polígonos dos GVEs
mapa_km_gve = mapa_unidades_saude.dissolve(by='GVE')

# Junta o formato do mapa com o resultado Cluster Hierarquico
mapa_km_gve = mapa_km_gve.merge(
    matriz_gve_cluster[['CLUSTER_EPIDEMICO_KM']], 
    left_index=True, 
    right_index=True, 
    how='left'
)

mapa_km_gve['LEGENDA'] = (
                            mapa_km_gve['CLUSTER_EPIDEMICO_KM'].astype(str).str.zfill(2)  + 
                            " - " + mapa_km_gve.index.astype(str).str.zfill(2)  +
                            " - " + mapa_km_gve['NOME_GVE'].astype(str)
                        )
#%% Plotagem do Mapa KM
fig, ax = plt.subplots(figsize=(14, 10))
mapa_km_gve.plot(
    ax=ax,
    column='CLUSTER_EPIDEMICO_KM',
    categorical=True,
    cmap=k4_cmap,          
    edgecolor='black',
    linewidth=0.5,
    legend=True,
)


legend_patches = []

# Loop para passar por cada um dos clusters
for i in range(n_k):
    dados_cluster = mapa_km_gve[mapa_km_gve['CLUSTER_EPIDEMICO_KM'] == i]
    lista_legendas = dados_cluster['LEGENDA'].tolist()
    lista_legendas.sort()    
    
    # Só faz algo se o cluster não estiver vazio
    if len(lista_legendas) > 0:
        
        # A primeira linha recebe o quadradinho com a cor do cluster
        patch_principal = mpatches.Patch(color=paleta_k4[i], label=lista_legendas[0])
        legend_patches.append(patch_principal)
        
        # As linhas seguintes recebem um quadradinho "invisível" (color='none')
        # Isso mantém o recuo do texto alinhado perfeitamente!
        for texto_linha in lista_legendas[1:]:
            patch_invisivel = mpatches.Patch(color=paleta_k4[i], label=texto_linha)
            legend_patches.append(patch_invisivel)
            
        # Adiciona uma linha totalmente vazia no final do cluster para dar um respiro visual
        # (Mas não adiciona depois do último cluster para não sobrar espaço à toa)
        if i < 3: 
            espaco_vazio = mpatches.Patch(color='none', label='')
            legend_patches.append(espaco_vazio)

# Adiciona a legenda ao gráfico (A mesma configuração de antes)
ax.legend(
    handles=legend_patches, 
    title="Perfil da Epidemia (Cluster K-Means)\n",
    loc='upper left', 
    bbox_to_anchor=(1.0, 1), # Mantém para fora do mapa
    prop={'family': 'Arial', 'size': 10}, 
    frameon=False, 
    labelspacing=0.3 # Deixa as linhas do mesmo cluster mais juntinhas
)
# Formata o título da legenda
plt.setp(ax.get_legend().get_title(), 
         fontproperties={'family': 'Arial', 'size': 10})

 # Códigos nos centróides
for idx, row in mapa_km_gve.iterrows():
     centroid = row.geometry.centroid
     if not centroid.is_empty:
         ax.annotate(
             text=str(idx), 
             xy=(centroid.x, centroid.y), 
             horizontalalignment='center', 
             fontsize=8, 
             color='black'
         )

ax.axis('off')
plt.tight_layout() # Garante que a legenda não seja cortada ao salvar a imagem
plt.show()

# %% Validacao do KMEANS
# ====== Formato TCC
# Configuração da figura com 4 painéis
fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
axes = axes.flatten()

# Garantir fundo branco total (Regra: sem preenchimento)
fig.patch.set_facecolor('white')

for i in range(4):
    # Ajuste do fundo de cada painel
    axes[i].set_facecolor('white')
    
    # Filtro dos dados do cluster
    gves_no_cluster = (matriz_gve_cluster['CLUSTER_EPIDEMICO_KM'] == i).values
    series_cluster_idx = matriz_gve[gves_no_cluster]
    
    # Desenha as linhas das GVEs (com a cor da paleta para o respectivo cluster)
    for s in series_cluster_idx.values:
        axes[i].plot(s, color=paleta_k4[i], linewidth=1)
        
    # Título do Painel: ALINHADO À ESQUERDA e em MAIÚSCULO (Regra Adicional)
    titulo_painel = f'CLUSTER {i} (N={len(series_cluster_idx)} GVES)'
    axes[i].set_title(titulo_painel.upper(), loc='left', fontsize=12, fontweight='bold', color=paleta_k4[i], fontfamily='Arial')
    
    # Rótulos dos Eixos (Arial, Tam 11, Preto)
    axes[i].set_ylabel('Casos Notificados', fontsize=11, color='black', fontfamily='Arial')
    axes[i].set_xlabel('Semana Epidemiológica', fontsize=11, color='black', fontfamily='Arial')
    
    
    # Formatação dos Ticks (Arial, Tam 10)
    axes[i].tick_params(axis='both', which='major', labelsize=10)
    for label in (axes[i].get_xticklabels() + axes[i].get_yticklabels()):
        label.set_fontname('Arial')

    # Regras da Tabela 8: Grades, Bordas e Eixos (1.5pt)
    axes[i].grid(False) # Não deve conter linhas de grade
    
    # Formatação das linhas dos eixos (Sólida, Preta, 1.5pt)
    axes[i].spines['top'].set_visible(False)    # Sem borda superior
    axes[i].spines['right'].set_visible(False)  # Sem borda direita
    axes[i].spines['bottom'].set_linewidth(1.5) # Eixo principal horizontal
    axes[i].spines['bottom'].set_color('black')
    axes[i].spines['left'].set_linewidth(1.5)   # Eixo principal vertical
    axes[i].spines['left'].set_color('black')

# Ajuste final de layout (Removido plt.suptitle para cumprir a Norma de "Não conter título principal")
plt.tight_layout()
plt.show()
