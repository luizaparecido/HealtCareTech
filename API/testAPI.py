from fastapi import FastAPI
import uvicorn



app = FastAPI()


@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/pagina")
async def read_page():
    return {"message": "This is a new page"}    

@app.get("/pagina_boba")
async def read_page():
    return {"message": "This is a new page"}   

def main():

    uvicorn.run(app, port=8000)


if __name__ == "__main__":
    main()