import os
import joblib
import xgboost as xgb
import numpy as np
from services.datasetService import dataset_completo


class DiagnosticoIA:
    
    def __init__(self, caminhoModelo: str):
        self.caminho_raiz = caminhoModelo
        self.HealthIA = xgb.XGBClassifier()
        self.HealthIA.load_model(os.path.join(self.caminho_raiz, 'modelo_HealthIA.json'))
        self.vetorizadortfidf = joblib.load(os.path.join(self.caminho_raiz, 'vetorizador_HealthIA.pkl'))
        self.encoderYPronto = joblib.load(os.path.join(self.caminho_raiz, 'encoderY_HealthIA.pkl'))
        
        # Lista simples com os nomes dos diagnósticos


    def predict_simples(self, sintomas):
        """
        Função simples para fazer predição
        """
        # Converter lista para string se necessário
        if isinstance(sintomas, list):
            sintomas_string = " ".join(sintomas)
        else:
            sintomas_string = sintomas
        
        # Vetorizar os sintomas
        sintomas_vetorizados = self.vetorizadortfidf.transform([sintomas_string])
        
        # Fazer predição
        predicao = self.HealthIA.predict(sintomas_vetorizados)
        pred_nome = self.encoderYPronto.classes_[int(predicao[0])] if hasattr(self.encoderYPronto, 'classes_') else str(predicao[0])

        LabelY = dataset_completo()["diagnostico"].astype(str).unique()
        label_pred = [LabelY[int(pred_nome[0])]]
        
        return label_pred


