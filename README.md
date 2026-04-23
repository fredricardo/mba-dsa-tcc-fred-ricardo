# Comparativo da Análise de Cluster no Agrupamento Geográfico da Epidemia de Dengue no Estado de São Paulo

**Autor:** Fred Ricardo de Souza Costa  
**Curso:** MBA em Data Science e Analytics – USP/Esalq (2026)

## Objetivo da Pesquisa
Esta pesquisa tem o objetivo de analisar os agrupamentos da incidência dos casos notificados de dengue no estado de São Paulo no ano de 2025. 
O estudo proporciona uma visão comparativa entre os algoritmos de clusterização não supervisionados não hierárquicos: **K-means** e **Dynamic Time Warping (DTW)**. 
O trabalho visa contornar a rigidez temporal imposta pela métrica de distância euclidiana do K-Means, utilizando o DTW para agrupar as séries temporais pela **morfologia das curvas de contágio**. Com isso, busca-se entender o comportamentoda dinâmica da epidemia de forma regionalizada (pelos Grupos de Vigilância Epidemiológica - GVE), 
fornecendo subsídios mais precisos para a formulação de políticas públicas de controle e prevenção.

# Comparativo da Análise de Cluster na Epidemia de Dengue (SP - 2025)

Este projeto realiza uma análise comparativa dos algoritmos de clusterização 
não supervisionados K-Means e Dynamic Time Warping (DTW) para entender o 
agrupamento geográfico das notificações de dengue no estado de São Paulo [1, 2].

## Pré-requisitos

Para executar o código, você precisará do Python instalado e das seguintes bibliotecas [3]:
*   `pandas`, `numpy`, `scipy`
*   `matplotlib`, `seaborn`, `plotly`
*   `geopandas`
*   `scikit-learn` (sklearn)
*   `tslearn`

Você pode instalá-las via pip:
`pip install pandas numpy scipy matplotlib seaborn plotly geopandas scikit-learn tslearn`

## 📂 Estrutura de Diretórios e Bases de Dados

Para que o código funcione corretamente, você deve organizar o repositório local com a seguinte estrutura de pastas relativas (o script busca os dados na pasta `../data/raw/`):

```text
├── data/
│   └── raw/
│       ├── DENGBR25.csv
│       ├── dengue25_se_vigilancia.csv
│       ├── pop_munic_ibge.csv
│       ├── municipio_tse_ibge.csv
│       ├── calendario_epidemiologico_2025.csv
│       └── SP_Municipios_2025.zip
├── src/
│   └── data_tcc_mba_dsa.py   # Código principal
└── README.md
````
O script espera que os dados estejam em uma pasta específica chamada `../data/raw/` em relação ao local de execução [4]. 

Faça o download das bases públicas abaixo e salve-as com os nomes indicados na pasta `data/raw/`:

1. **Base SINAN - Agravos de Notificação (Dengue)**
   *   **Onde obter:** Portal de Dados Abertos do Governo Federal ([Acessar link](https://dados.gov.br/dados/conjuntos-dados/arboviroses-dengue)) [5].
   *   **Nome do arquivo:** `DENGBR25.csv` [4].
2. **Dados Estatísticos CVE**
   *   **Onde obter:** Portal do Centro de Vigilância Epidemiológica ([Acessar link](https://www.saude.sp.gov.br/cve-centro-de-vigilancia-epidemiologica-prof.-alexandre-vranjac/areas-de-vigilancia/doencas-de-transmissao-por-vetores-e-zoonoses/arboviroses-urbanas/dengue-dados-estatisticos)) [6].
   *   **Nome do arquivo:** `dengue25_se_vigilancia.csv` [7].
3. **Códigos oficiais de UF e Municípios**
   *   **Onde obter:** Portal de Dados Abertos do TSE/IBGE ([Acessar link](https://dadosabertos.tse.jus.br/dataset/codigos-oficiais-de-uf-e-municipios-segundo-o-tse-e-o-ibge)) [8].
   *   **Nome do arquivo:** `municipio_tse_ibge.csv` [9].
4. **População Estimada (IBGE)**
   *   **Onde obter:** Portal do IBGE (Necessário baixar e tratar no formato CSV) [7].
   *   **Nome do arquivo:** `pop_munic_ibge.csv` [7].
5. **Calendário Epidemiológico 2025**
   *   **Onde obter:** Portal SINAN Ministério da Saúde ([Acessar link](https://portalsinan.saude.gov.br/calendario-epidemiologico/43-institucional/175-calendario-epidemiologico-2025)) [10].
   *   **Nome do arquivo:** `calendario_epidemiologico_2025.csv` [9].
6. **Malha Municipal de São Paulo**
   *   **Onde obter:** Portal de Geociências do IBGE ([Acessar link](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html)) [11].
   *   **Nome do arquivo:** `SP_Municipios_2025.zip` [9].

## Como Executar

Com as bibliotecas instaladas e as bases de dados posicionadas corretamente na pasta `../data/raw/`, basta executar o script principal:

```bash

python data_tcc_mba_dsa.py