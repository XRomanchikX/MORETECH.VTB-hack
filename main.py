from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# Импортируйте ваш класс
from screenCV import InitialScreening  # замените на правильный путь

app = FastAPI(title="CV Screening API", version="1.0.0")

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В production укажите конкретные origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CVRequest(BaseModel):
    vacancyPath: str
    cvPath: str

class CVResponse(BaseModel):
    score: float
    verdict: str

@app.post("/api/process-cv", response_model=CVResponse)
async def process_cv(request: CVRequest):
    try:
        # Создаем экземпляр класса и обрабатываем
        screening = InitialScreening(
            PathToVacancy=request.vacancyPath,
            PathToCV=request.cvPath
        )
        
        score = screening.check_cv()  # возвращает float, например 75.3
        verdict = 'approved' if score >= 50.0 else 'rejected'
        
        return CVResponse(score=score, verdict=verdict)
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Файл не найден: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)