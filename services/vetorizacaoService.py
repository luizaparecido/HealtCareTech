from sklearn.feature_extraction.text import TfidfVectorizer
from services.datasetService import dataset_completo
from sklearn.preprocessing import LabelEncoder


# 1 - chamar o framework que teo o TF-IDF
# 2 - Instanciar o modelo de vetorizacao
# 3 - Pegar os datos e vetorizar



def vetorizacao():
    tfidf = TfidfVectorizer()

    df = dataset_completo()
    X = df["sintomas"].astype(str)

    tfidf.fit(X)
   
    x_tfidf = tfidf.transform(X)

    return x_tfidf 
 
def encode_Y():
    label_encoder = LabelEncoder()
    df = dataset_completo()
    Y = df['diagnostico'].astype(str) 

    y_encoded = label_encoder.fit_transform(Y)

    return y_encoded