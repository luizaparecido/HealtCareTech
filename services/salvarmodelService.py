from services.vetorizacaoService import vetorizador, encode_Y
from services.treinamentoService import treinar_modelo
import pickle
import os


# Salvar o vetorizador
# Salvar o Encoder (Label Y)
# Salvar o modelo treinado



def salvar_vetorizador():
    """
    Salvar o vetorizador como arquivo no diretorio model      
    
    """
    vetorizador_criado = vetorizador()
    # salvar o vetorizador
    caminho_arquivo = 'model/vetorizador_HealthIA.pkl'
    # Check if directory exists, create it if it doesn't
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)

    with open(caminho_arquivo, "wb") as f:
        pickle.dump(vetorizador_criado, f)
        print("Vetorizador salvo com sucesso!")


def salvar_encoderY():
    """
    Salvar o encoder Y como arquivo no diretorio model
    """
    encoderY_criado = encode_Y()
    # salvar o encoder Y
    caminho_arquivo = 'model/encoderY_HealthIA.pkl'
    # Check if directory exists, create it if it doesn't
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)

    with open(caminho_arquivo, "wb") as f:
        pickle.dump(encoderY_criado, f)
        print("Encoder Y salvo com sucesso!")

def salvar_modelo():

    """
    Salvar o modelo treinado como arquivo no diretorio model
    """
    HealthIA = treinar_modelo()
    # salvar o modelo
    caminho_arquivo = 'model/modelo_HealthIA.json'
    # Check if directory exists, create it if it doesn't
    os.makedirs(os.path.dirname(caminho_arquivo), exist_ok=True)

    HealthIA.save_model(caminho_arquivo)
    print("Modelo salvo com sucesso!")
