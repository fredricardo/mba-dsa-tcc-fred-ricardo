# Agrupamento geográfico da epidemia da dengue em São Paulo por K-Means e Dynamic Time Warping

**Autor:** Fred Ricardo de Souza Costa  
**Orientador:** Fabiano Sannino  
**Curso:** MBA em Data Science e Analytics – USP/Esalq (2026)  
**Código:** [`src/data_tcc_mba_dsa.py`](src/data_tcc_mba_dsa.py)

---

## Objetivo

Analisar os agrupamentos da incidência dos casos notificados de dengue no Estado de São Paulo no ano de 2025, fornecendo uma visão comparativa entre dois algoritmos de clusterização não supervisionados não hierárquicos: **K-Means** e **Dynamic Time Warping (DTW)**.

O estudo utiliza fontes públicas do SINAN cruzadas com informações geográficas territoriais do estado para traçar uma visão espacial do comportamento da epidemia, com abordagem exploratória — sem fins preditivos para observações fora da amostra.

---

## Contexto epidemiológico

Em 2025, o Centro de Vigilância Epidemiológica "Prof. Alexandre Vranjac" (CVE) registrou **1.992.259 notificações** no Estado de São Paulo, com **859.068 casos confirmados** de dengue, sobrecarregando o sistema público de saúde. A análise utilizou **882.084 registros confirmados** extraídos da base SINAN (≈53% do total de notificações do estado), filtrados pelos critérios:

- `SG_UF_NOT = 35` (Estado de São Paulo, código IBGE)
- `CLASSI_FIN >= 10` (Dengue Clássica=10, com Sinais de Alarme=11, Grave=12)

As análises foram direcionadas aos **28 Grupos de Vigilância Epidemiológica (GVE)**, unidade organizacional regional de saúde com papel fundamental no monitoramento contínuo das doenças no estado.

---

## Metodologia

### Dados e pré-processamento

A série temporal foi definida pelo acúmulo de notificações na semana epidemiológica por GVE no ano de 2025, usando o campo `SEM_PRI` (semana epidemiológica dos primeiros sintomas/diagnósticos).

Para compatibilizar escalas populacionais entre os GVEs, calculou-se a **taxa de incidência** (casos por 100 mil habitantes por semana epidemiológica), obtendo-se a matriz padronizada **GVE × Semana Epidemiológica (28×53)** via **Z-Score**. Essa normalização foi necessária sobretudo pela sensibilidade do DTW a ruídos e valores atípicos que podem distorcer as medidas de similaridade.

### K-Means

- Distância **euclidiana** entre semanas epidemiológicas
- Centroide baseado na **média aritmética** de cada cluster
- Limitação: alinha os dados de forma rígida ao calendário epidemiológico — GVEs com surtos de morfologia idêntica, mas defasados por semanas, são separados em clusters distintos

### Dynamic Time Warping (DTW)

- Clusterização pela **morfologia das curvas** de contágio
- Baseado nos estudos de Sakoe e Chiba (1978) sobre otimização por programação dinâmica
- Centroide extraído pelo método **DTW Barycenter Averaging (DBA)**, que representa a trajetória da epidemia no grupo — não a média aritmética de casos em um período
- Supera a defasagem temporal: GVEs com mesma dinâmica de transmissão são agrupados independentemente de quando o surto ocorreu

### Definição do número de clusters

Para ambos os métodos:
- **Método Elbow** (cotovelo): WCSS para K-Means e soma dos custos de alinhamento DTW
- **Coeficiente de Silhueta**: validação da qualidade dos agrupamentos
- Ambas as métricas indicaram k=2 como ponto ótimo, porém dois grupos se mostraram insuficientes para capturar a granularidade regional dos 28 GVEs. Adotou-se **k=4** como ponto de equilíbrio, mantendo comparabilidade direta entre os dois modelos.

### Comparação entre métodos

O **Adjusted Rand Index (ARI)** retornou escore de **0,0493** ao comparar os agrupamentos K-Means e DTW para k=4, comprovando matematicamente concordância quase nula — a rigidez temporal euclidiana e a métrica elástica do DTW agrupam os 28 GVEs sob critérios fundamentalmente divergentes.

### Implementação

```python
# Bibliotecas principais
from sklearn.cluster import KMeans                    # scikit-learn 1.3.0
from sklearn.metrics import silhouette_score, adjusted_rand_score
from tslearn.clustering import TimeSeriesKMeans        # tslearn 0.6.0
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.metrics import cdist_dtw

# Parâmetros de controle de convergência (ambos os modelos)
random_state = 42   # Determinismo das inicializações
max_iter     = 300
tol          = 1e-4
n_init       = 10   # K-Means
n_init       = 1    # DBA-KMeans
```

Todo o código está disponível publicamente em:  
**https://github.com/fredricardo/mba-dsa-tcc-fred-ricardo**

---

## Pré-requisitos

Python 3.x com as seguintes bibliotecas (ver [`requirements.txt`](requirements.txt)):

| Grupo | Bibliotecas |
|---|---|
| Dados | `pandas`, `numpy`, `scipy` |
| Visualização | `matplotlib`, `seaborn`, `plotly` |
| Geoespacial | `geopandas`, `shapely` |
| Machine Learning | `scikit-learn >= 1.3.0` |
| Séries Temporais | `tslearn >= 0.6.0` |

Instalação:

```bash
pip install -r requirements.txt
```

Ou dentro de um notebook:

```python
%pip install -r requirements.txt
```

---

## Estrutura de diretórios

```text
├── data/
│   └── raw/
│       ├── DENGBR25.csv                        # Base SINAN – notificações 2025
│       ├── dengue25_se_vigilancia.csv          # Dados estatísticos CVE (divisões organizacionais)
│       ├── pop_munic_ibge.csv                  # População estimada por município (IBGE)
│       ├── municipio_tse_ibge.csv              # Códigos oficiais UF/municípios TSE+IBGE
│       ├── calendario_epidemiologico_2025.csv  # Calendário epidemiológico SINAN 2025
│       └── SP_Municipios_2025.zip              # Malha Municipal IBGE 2025 (formato shapefile)
├── src/
│   └── data_tcc_mba_dsa.py                     # Script principal
├── requirements.txt
└── README.md
```

O script busca os dados em `../data/raw/` relativo ao local de execução.

---

## Bases de dados

Faça o download das bases públicas abaixo e salve-as com os nomes indicados em `data/raw/`:

| # | Base | Fonte | Nome do arquivo |
|---|---|---|---|
| 1 | SINAN – Agravos de Notificação (Dengue) | [Portal de Dados Abertos do Governo Federal](https://dados.gov.br/dados/conjuntos-dados/arboviroses-dengue) | `DENGBR25.csv` |
| 2 | Dados Estatísticos CVE | [CVE – Dengue Dados Estatísticos](https://www.saude.sp.gov.br/cve-centro-de-vigilancia-epidemiologica-prof.-alexandre-vranjac/areas-de-vigilancia/doencas-de-transmissao-por-vetores-e-zoonoses/arboviroses-urbanas/dengue-dados-estatisticos) | `dengue25_se_vigilancia.csv` |
| 3 | Códigos oficiais UF e municípios | [Portal de Dados Abertos TSE/IBGE](https://dadosabertos.tse.jus.br/dataset/codigos-oficiais-de-uf-e-municipios-segundo-o-tse-e-o-ibge) | `municipio_tse_ibge.csv` |
| 4 | População Estimada (IBGE) | Portal IBGE (tratar e exportar como CSV) | `pop_munic_ibge.csv` |
| 5 | Calendário Epidemiológico 2025 | [Portal SINAN – Ministério da Saúde](https://portalsinan.saude.gov.br/calendario-epidemiologico/43-institucional/175-calendario-epidemiologico-2025) | `calendario_epidemiologico_2025.csv` |
| 6 | Malha Municipal São Paulo 2025 | [Geociências IBGE](https://www.ibge.gov.br/geociencias/organizacao-do-territorio/malhas-territoriais/15774-malhas.html) | `SP_Municipios_2025.zip` |

**Notas sobre a base SINAN:**
- O campo `ID_MUNICIP` não considera o último dígito conforme código do IBGE
- O campo `SEM_PRI` (semana dos primeiros sintomas) foi adotado em vez de `SEM_NOT` (semana da notificação)
- A classificação final `CLASSI_FIN` aceita os valores 10 (Dengue Clássica), 11 (com Sinais de Alarme) e 12 (Grave) como casos confirmados
- Os campos `COUFINF`/`COMUNINF` não foram utilizados, pois representam probabilidade de fonte de infecção anotada por outros estados com procedimentos diferenciados de notificação
- Os dados do SINAN podem apresentar atrasos de notificação por fatores externos ao processo de registro nas unidades de atendimento

---

## Como executar

Com as dependências instaladas e os arquivos posicionados em `data/raw/`, execute:

```bash
cd src
python data_tcc_mba_dsa.py
```

O script produz sequencialmente:

1. **Curva epidemiológica global** – frequência absoluta de notificações por semana epidemiológica (campo `SEM_PRI`)
2. **Tendência com média móvel de 7 dias** – o pico global de transmissão ocorreu entre março e abril de 2025
3. **Mapas das divisões organizacionais de saúde** – DRS, GVE e Regiões de Saúde sobre a malha municipal
4. **Notificações absolutas por GVE** – curvas individuais dos 28 grupos com destaque para a média estadual
5. **Matriz padronizada (Z-Score)** – heatmap GVE × Semana Epidemiológica (28×53)
6. **K-Means** – Elbow, Silhueta, centroides por média aritmética, e mapa dos clusters por GVE
7. **DTW** – Elbow, Silhueta, morfologias DBA, mapa dos clusters por GVE
8. **ARI** – métrica de concordância entre os dois agrupamentos

---

## Limitações

- Estudo limitado ao ano de 2025; temporadas epidêmicas anteriores não foram verificadas
- Abordagem exploratória sem fins preditivos para observações fora da amostra
- Variável única geográfica: disposição morfológica das ondas epidêmicas nas regiões — aspectos climáticos, geológicos e índices de infestação vetorial não foram considerados, mas podem ser correlacionados em estudos complementares
- Sem validação externa dos agrupamentos contra desfechos independentes (óbitos, internações)

---

## Dúvidas e contribuições

Caso necessite de mais informações sobre o estudo, os dados ou o código, abra uma [issue no GitHub](https://github.com/fredricardo/mba-dsa-tcc-fred-ricardo/issues) que terei o prazer de responder.

---

## Referências

- Centro de Vigilância Epidemiológica [CVE]. 2026. *Boletim Epidemiológico – Arboviroses SE 01-53/2025.*
- Sakoe, H.; Chiba, S. 1978. Dynamic programming algorithm optimization for spoken word recognition. *IEEE Transactions on Acoustics, Speech, and Signal Processing* 26(1): 43–49.
- Petitjean, F.; Ketterlin, A.; Gancarski, P. 2011. A global averaging method for dynamic time warping, with applications to clustering. *Pattern Recognition* 44(3): 678–693.
- Hubert, L.; Arabie, P. 1985. Comparing partitions. *Journal of Classification* 2: 193–218.
- Kobylin, O.; Lyashenko, V. 2020. Time Series Clustering Based on the K-Means Algorithm. *Journal La Multiapp* 1: 1–7.
- Fávero, L.P.; Belfiore, P. 2024. *Manual de análise de dados: estatística e machine learning com Excel, SPSS, Stata, R e Python.* 2ed. LTC.
- Gouveia et al. 2025. Unraveling regional variability in Dengue outbreaks in Brazil: leveraging the Moving Epidemics Method (MEM) and climate data to optimize vector control strategies. *PLOS Neglected Tropical Diseases* 19(6): e0013175.
- Kirpich et al. 2025. Clustering-based methodology for comparing multi-characteristic epidemiological dynamics with application to COVID-19 epidemiology in Europe. *Royal Society Open Sci.* 12(9): 250440.
