from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from model.predict import predict_pipeline
from model.predict import __version__ as model_version

app = FastAPI()

class PredictionOut(BaseModel):
    sport_type: str
    confidence: float

@app.get('/')
def home():
    return {'health_check': 'OK', 'model_version': model_version}

@app.post('/predict', response_model=PredictionOut)
async def asyncpredict(image: UploadFile = File()):
    prediction, confidence = await predict_pipeline(image)
    return {"sport_type": prediction, "confidence": confidence}

# To run: uvicorn main:app --reload

