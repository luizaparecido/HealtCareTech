

from services.datasetService import dataset_completo
from services.vetorizacaoService import vetorizacao, encode_Y


def prints_dataset():
  df = dataset_completo()
  return df

def prints_vetorizacao():
  x_tfidf = vetorizacao()
  print(x_tfidf)

def prints_encode_Y():
  y_encoded = encode_Y()
  print(y_encoded)

if __name__ == "__main__":
  prints_encode_Y()


