from API.sintomasAPI import router as sintomas_router
from fastapi import FastAPI


app = FastAPI(title="HealthIA API", description="API para predição de diagnósticos médicos com base em sintomas.", version="1.0.0")
app.include_router(sintomas_router,  tags=["Sintomas"])