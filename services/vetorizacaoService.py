from sklearn.feature_extraction.text import TfidfVectorizer
from services.datasetService import dataset_completo
from sklearn.preprocessing import LabelEncoder



# 1 - chamar o framework que tem o TF-IDF - ok
# 2 - instanciar o modelo de vetorização - ok
# 3 - Pegar os dados e vetorizar


# Vetorizador
def vetorizador():
    """
    Funcao que cria um vetorizador TF-IDF e o ajusta aos dados de entrada.

    Returns:
        Vetorizador ajustado com os dados pronto para ser usado.
    """
    tfidf = TfidfVectorizer()

    df = dataset_completo()
    
    X = df["sintomas"].astype(str)

    tfidf.fit(X)
    
    return tfidf

#Vetorizador vetorizando os Dados
# retorno Dados Vetorizados
def vetorizacao():
    tfidf = TfidfVectorizer()

    df = dataset_completo()
    X = df["sintomas"].astype(str)

    tfidf.fit(X)

    X_tfidf = tfidf.transform(X)

    return X_tfidf

def encode_Y():
    """
    Função para codificar a variável alvo (Y) usando Label Encoding.

    Return:
    Y_encoded: array
        Array com os rótulos codificados.
    """
    label_encoder = LabelEncoder()
    df = dataset_completo()

    Y = df["diagnostico"].astype(str)

    Y_encoded = label_encoder.fit_transform(Y)

    return Y_encoded





