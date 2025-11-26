

from services.datasetService import dataset_completo
from services.vetorizacaoService import vetorizacao, encode_Y
from services.treinamentoService import acuracia_modelo


def prints_dataset():
  df = dataset_completo()
  return df

def prints_vetorizacao():
  x_tfidf = vetorizacao()
  print(x_tfidf)

def prints_encode_Y():
  y_encoded = encode_Y()
  print(y_encoded)

def acuracia_modelo_print():

    acuracia = acuracia_modelo()
    print(f"Acurácia do modelo: {acuracia:.2f}%")


if __name__ == "__main__":
  
  acuracia_modelo_print()





