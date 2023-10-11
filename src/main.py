from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import datetime
import lightgbm as lgb

# Create a FastAPI app
app = FastAPI()

# Create a Pydantic model to validate request data
class ModelData(BaseModel):
    item_id: int
    store_id: int
    date: str


class ForecastSales(BaseModel):
    steps: int


# Project description
project_description = """
This is a project that demonstrates a machine learning model.

- **Endpoint 1** /sales/stores/items/: Predict something item_id, store_id and date.
- **Endpoint 2** /sales/national/ : Another endpoint for a forecasting the events.
- **Endpoint 3** /docs : Check documentation.
"""

# GitHub repository link
github_repo = "Link to the GitHub repository: [Your GitHub Repo URL]"


@app.get("/", summary="Basic Info of the project")
async def basic_info():
    return Response(content=f"Processed: {project_description} - {github_repo}", media_type="text/plain")


@app.get("/health", summary="Greeting of the project")
async def welcome_message():
    return Response(content=f"message: Welcome to the Sales Prediction Project", media_type="text/plain")


# Define a POST endpoint to greet a person
@app.post("/sales/stores/items/")
async def predict_sales(model: ModelData):
    try:
        prediction_model = pickle.load(open('prediction_model.pkl', 'rb'))

        # Convert the date string to a datetime object
        datetime_ = datetime.datetime.strptime(model.date, '%Y-%m-%d')

        df = pd.DataFrame([{
            "item_id": model.item_id,
            "store_id": model.store_id,
            "day": datetime_.date().day,
            "month": datetime_.date().month,
            "year": datetime_.date().year
        }])


        # prediction_model.predict
        res = list(prediction_model.predict(df))
        return {"The output is ": [round(i,2) for i in res]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/sales/national/")
async def predict_sales(model: ForecastSales):
    try:
        prediction_model = pickle.load(open('forecast_model.pkl', 'rb'))
        # prediction_model.predict
        res = list(prediction_model.forecast(model.steps))
        return {"The output is ": [round(i,2) for i in res]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)