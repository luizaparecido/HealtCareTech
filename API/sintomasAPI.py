from fastapi import APIRouter
from services.executarService import DiagnosticoIA



router = APIRouter()
diagnosticoIA = DiagnosticoIA(caminhoModelo='model')


@router.get("/")
async def raiz():
    return {"Message": "Welcome to HealthIA API"}


#usuario envia os sintomas via query params
@router.get("/predict/")
async def predict(sintomas: str):
    """
    Endpoint para predição de diagnóstico com base em sintomas fornecidos via query params.

    Parâmetros:
    sintomas (str): Uma string contendo os sintomas separados por vírgulas.

    Retorna:
    dict: Um dicionário contendo o diagnóstico previsto.
    """
    # Aqui você pode adicionar a lógica para carregar o modelo treinado e fazer a predição
    # Por enquanto, vamos retornar uma resposta simulada
    sintomas_list = sintomas.split(',')
    # Simulação de predição
    diagnostico_previsto = diagnosticoIA.predict_simples(sintomas_list)

    return {
        "sintomas": sintomas_list,
        "diagnostico_previsto": diagnostico_previsto
    }
