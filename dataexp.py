import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    pairwise_distances, classification_report, confusion_matrix,
    accuracy_score, mean_absolute_error, mean_squared_error, max_error, r2_score
)
import numpy as np

# Definir função de plotagem no Streamlit
def plot(xValues, yValues, annotate, number_formatting, title, xLabel, yLabel):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=xValues, y=yValues, hue=xValues, palette="viridis", dodge=False)
    
    if annotate:
        for p in ax.patches:
            ax.annotate(format(p.get_height(), number_formatting),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', xytext=(0, 5),
                        textcoords='offset points', fontsize=10)

    plt.title(title, fontsize=16)
    plt.xlabel(xLabel, fontsize=12)
    plt.ylabel(yLabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(plt)

# Simular dados (substitua pelo carregamento do seu dataset real)
df=df = pd.read_csv('datatran_tratado.csv')

df_new = df.copy() # cria uma cópia do df original

df_new['data_inversa'] = pd.to_datetime(df_new['data_inversa']) # Formata a data para evitar inconsistências
df_new['horario'] = pd.to_datetime(df_new['horario'], format='%H:%M:%S').dt.time # Formata o horário para evitar inconsistências

df_new['month'] = df_new['data_inversa'].dt.month # Gera uma nova coluna com o dia em que o acidente aconteceu
df_new['day'] = df_new['data_inversa'].dt.day # Gera uma nova coluna com o dia em que o acidente aconteceu
df_new['hour'] = pd.to_datetime(df_new['horario'], format='%H:%M:%S').dt.hour # Gera uma nova coluna com a hora em que o acidente aconteceu

df_new['km'] = df_new['km'].str.replace(',', '.').astype(float) # Formata a coluna 'km' convertendo todos os seus valores para float

# Usa a técnica de Enconding para variáveis categóricas ordinais para transformar as strings object dos dias da semana para números int32
le = LabelEncoder()

colunas_para_label_encode = ['uf', 'causa_acidente', 'tipo_acidente',
    'sentido_via','classificacao_acidente', 'tracado_via', 'condicao_metereologica',
    'tipo_pista', 'regional', "dia_semana", "fase_dia", "br", "uso_solo"]

for coluna in colunas_para_label_encode:
  df_new[coluna] = le.fit_transform(df_new[coluna])

# Selecionar características e alvo
X = df_new[['causa_acidente', 'tipo_acidente', 'condicao_metereologica', 'veiculos', 'tracado_via', 'dia_semana', 'fase_dia', 'tipo_pista', 'sentido_via', 'uso_solo',
        'pessoas', 'latitude', 'longitude', 'br', 'km']]
y = df_new['classificacao_acidente']

print(y.value_counts())

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

X = X_resampled
y = y_resampled

print(y.value_counts())

# Separar em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Lista de modelos a serem testados
# models = {
#     'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3),
#     'RandomForestClassifier': RandomForestClassifier(),
#     'DecisionTreeClassifier': DecisionTreeClassifier()
# }

# # Dicionário para armazenar os resultados
# results = {}

# # Testar cada modelo
# for model_name, model in models.items():
#     # Criar um pipeline com escalação e o modelo
#     pipeline = Pipeline([
#         ('scaler', StandardScaler()),  # Escalonamento dos dados
#         ('classifier', model)
#     ])

#     # Treinar o modelo
#     pipeline.fit(X_train, y_train)

#     # Fazer previsões
#     y_pred = pipeline.predict(X_test)

#     # Avaliar o modelo
#     report = classification_report(y_test, y_pred, output_dict=True)
#     results[model_name] = report

# # Exibir os resultados
# for model_name, report in results.items():
#     print(f"Resultados para {model_name}:")
#     print(f"Acurácia: {report['accuracy']:.2f}")
#     print(f"Precisão: {report['weighted avg']['precision']:.2f}")
#     print(f"Recall: {report['weighted avg']['recall']:.2f}")
#     print(f"F1-Score: {report['weighted avg']['f1-score']:.2f}\n")

model = RandomForestClassifier()

# Definindo o K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# # Calculando as scores de validação cruzada
# scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# print(f'Scores de Validação Cruzada: {scores}')
# print(f'Média da Acurácia: {scores.mean()}')

model.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# Sidebar para seleção de visualização
st.sidebar.title("Analise exploratoria")
visualizacao = st.sidebar.selectbox("Escolha a visualização de dados:", 
                                    ['Acidentes por Condição Meteorológica', 
                                     'Acidentes com Vítimas Fatais por Condição Meteorológica',
                                     'Proporção de Acidentes por Classificação',
                                     'Causas de Acidentes Fatais',
                                     'Quantidade Média de Veículos por Gravidade do Acidente',
                                     'Acidentes por Dia da Semana',
                                     'Causas de Acidentes por Dia da Semana',
                                     'Acidentes Graves por Regiao',
                                     'Hipótese 2'])

# Lógica para exibir os gráficos com base na escolha do usuário
if visualizacao == 'Acidentes por Condição Meteorológica':
    contagem_acidentes_por_condicoes = df['condicao_metereologica'].value_counts()
    proporcao_acidentes_por_condicoes = df['condicao_metereologica'].value_counts(normalize=True) * 100
    plot(xValues=contagem_acidentes_por_condicoes.index, yValues=proporcao_acidentes_por_condicoes, 
         annotate=True, number_formatting='0.5f', title="Quantidade de Acidentes por Condições Meteorológicas", 
         xLabel="Condições Meteorológicas", yLabel="Proporção (%)")

elif visualizacao == 'Acidentes com Vítimas Fatais por Condição Meteorológica':
    df_adversas = df[df['condicao_metereologica'] != 'Céu Claro']
    df_vitimas_fatais = df_adversas[df_adversas['classificacao_acidente'] == 'Com Vítimas Fatais']
    acidentes_vitimas_fatais = df_vitimas_fatais.groupby('condicao_metereologica').size().sort_values(ascending=False)
    plot(xValues=acidentes_vitimas_fatais.index, yValues=acidentes_vitimas_fatais, annotate=True, 
         number_formatting='0.1f', title="Condições Meteorológicas e Vítimas Fatais", 
         xLabel="Condições Meteorológicas", yLabel="Quantidade de Acidentes Fatais")

elif visualizacao == 'Proporção de Acidentes por Classificação':
    contagem_acidentes_classificacao = df['classificacao_acidente'].value_counts()
    proporcao_acidentes_classificacao = df['classificacao_acidente'].value_counts(normalize=True) * 100
    plot(xValues=contagem_acidentes_classificacao.index, yValues=proporcao_acidentes_classificacao, 
         annotate=True, number_formatting='0.5f', title="Proporção de Acidentes Leves, Graves e Fatais", 
         xLabel="Classificação do Acidente", yLabel="Proporção (%)")

elif visualizacao == 'Causas de Acidentes Fatais':
    acidentes_fatais = df[df['classificacao_acidente'] == 'Com Vítimas Fatais']
    causas_fatais = acidentes_fatais.groupby('causa_acidente').size().reset_index(name='total_acidentes_fatais')
    causas_fatais = causas_fatais.sort_values(by='total_acidentes_fatais', ascending=False)
    plot(xValues=causas_fatais['causa_acidente'].head(5), yValues=causas_fatais['total_acidentes_fatais'].head(5), annotate=True, number_formatting='0.1f', title="Causas Associadas a Acidentes Fatais", xLabel="Causas", yLabel="Quantidade de Acidentes Fatais")

elif visualizacao == 'Quantidade Média de Veículos por Gravidade do Acidente':
    gravidade_veiculos = df.groupby('classificacao_acidente')['veiculos'].mean().reset_index()
    plot(xValues=gravidade_veiculos['classificacao_acidente'], yValues=gravidade_veiculos['veiculos'], 
         annotate=True, number_formatting='0.3f', title="Quantidade Média de Veículos Envolvidos e Gravidade do Acidente", 
         xLabel="Classificação do Acidente", yLabel="Média de Veículos Envolvidos")

elif visualizacao == 'Acidentes por Dia da Semana':
    acidentes_por_dia = df['dia_semana'].value_counts().reset_index()
    acidentes_por_dia.columns = ['dia_semana', 'total_acidentes']
    dias_ordenados = ['segunda-feira', 'terça-feira', 'quarta-feira', 'quinta-feira', 'sexta-feira', 'sábado', 'domingo']
    acidentes_por_dia['dia_semana'] = pd.Categorical(acidentes_por_dia['dia_semana'], categories=dias_ordenados, ordered=True)
    acidentes_por_dia = acidentes_por_dia.sort_values('dia_semana')
    plot(xValues=acidentes_por_dia['dia_semana'], yValues=acidentes_por_dia['total_acidentes'], 
         annotate=True, number_formatting='0.1f', title="Número de Acidentes por Dia da Semana", 
         xLabel="Dia da Semana", yLabel="Total de Acidentes")

elif visualizacao == 'Causas de Acidentes por Dia da Semana':
    causas_dias = df.groupby(['dia_semana', 'causa_acidente']).size().reset_index(name='contagem').sort_values(by='contagem', ascending=False)
    dias_da_semana = causas_dias['dia_semana'].unique()

    for dia_da_semana in dias_da_semana:
        dia_causas = causas_dias[causas_dias['dia_semana'] == dia_da_semana].head(5)
        plot(xValues=dia_causas['causa_acidente'], yValues=dia_causas['contagem'], 
             annotate=True, number_formatting='0.1f', 
             title=f"Causas de Acidentes no(a) {dia_da_semana}", 
             xLabel="Causas", yLabel="Contagem")

elif visualizacao == 'Acidentes Graves por Regiao':
    df['gravidade_acidente'] = df['classificacao_acidente'].apply(lambda x: 'Grave' if x in ['Com Vítimas Fatais', 'Com Vítimas Feridas'] else 'Leve')
    acidentes_por_regional = df.groupby(['regional', 'gravidade_acidente']).size().reset_index(name='contagem')
    regionais_graves = acidentes_por_regional[acidentes_por_regional['gravidade_acidente'] == 'Grave'].sort_values(by='contagem', ascending=False)
    plot(xValues=regionais_graves['regional'], yValues=regionais_graves['contagem'], annotate=True, number_formatting='0.1f', title="Regiões com Mais Acidentes Graves", xLabel="Regionais", yLabel="Quantidade de Acidentes Graves")

elif visualizacao == 'Hipótese 2':
    y_pred = model.predict(X_test)  # Substitua por seu conjunto de teste
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.title('Matriz de Confusão')
    plt.show()
    st.pyplot(plt)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns  # Substitua se necessário

    plt.figure()
    plt.title('Importância das Características')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), features[indices], rotation=90)
    plt.xlim([-1, X.shape[1]])
    plt.show()

    st.pyplot(plt)
    
