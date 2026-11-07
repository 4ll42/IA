import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
import io

URL_DADOS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQsH48IL57JPFPhacRy18uelEB_AIDUdPYz5LSkdYnv15m_S5ZVDDoYZ4A-TJjj-yyEY3xnMabZTUvz/pub?output=csv"

NOME_ARQUIVO_MODELO = 'modelo_desempenho.pkl'

def treinar_modelo():
    print(f"Baixando dados de {URL_DADOS}")
    data = pd.read_csv(URL_DADOS)

    print("--- Dados Carregados ---")
    print(data.head())

    print("--- Preparando dados para o treino ---")
    features = ['horas_estudo', 'faltas', 'nota_p1']
    target = 'resultado'
    X = data[features]
    Y = data[target]

    modelo = KNeighborsClassifier(n_neighbors=3)

    modelo.fit(X, Y)

    print(f"--- Modelo Treinado! Classes: {modelo.classes_} ---")

    joblib.dump(modelo, NOME_ARQUIVO_MODELO)

    print(f"--- Modelo salvo com sucesso em '{NOME_ARQUIVO_MODELO}' ---")

if __name__ == "__main__":
    treinar_modelo()