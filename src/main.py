from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import datetime
import lightgbm as lgb

# Create a FastAPI app
app = FastAPI(reload=True)

# Create a Pydantic model to validate request data
class ModelData(BaseModel):
    item_id: int
    store_id: int
    date: str


# class ForecastSales(BaseModel):
    # steps: int


# Project description
project_description = """
This is a project that demonstrates a machine learning model.

- **Endpoint 1** /sales/stores/items/: Predict something item_id, store_id and date.
- **Endpoint 2** /sales/national/ : Another endpoint for a forecasting the events.
- **Endpoint 3** /docs : Check documentation.
"""

# GitHub repository link
github_repo = "https://github.com/aibarna/basnet_aibarna-24585717-ML_AT2"


@app.get("/", summary="Basic Info of the project")
async def basic_info():
    return Response(content=f"Processed: {project_description} - {github_repo}", media_type="text/plain")


@app.get("/health", summary="Greeting of the project")
async def welcome_message():
    return Response(content=f"message: Welcome to the Sales Prediction Project", media_type="text/plain")


@app.get("/sales/stores/items/" ,summary="Predict revenue for item and store on date ")
async def predict_sales(item_id: int, date:str, store_id:int):
    """_summary_

    Args:
        date:str: format should be similar to 2023-01-01
        store_id: int: 1
        item_id:int: 1

    Raises:
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
    try:
        prediction_model = pickle.load(open('prediction_model.pkl', 'rb'))

        # Convert the date string to a datetime object
        datetime_ = datetime.datetime.strptime(date, '%Y-%m-%d')

        df = pd.DataFrame([{
            "item_id": item_id,
            "store_id": store_id,
            "day": datetime_.date().day,
            "month": datetime_.date().month,
            "year": datetime_.date().year
        }])


        # prediction_model.predict
        res = list(prediction_model.predict(df))
        str_val = ""
        for i in res:
            str_val += " " + str(round(i,2))

        return Response(content=f"The output is for item {item_id} in store {store_id} on {date} is {str_val}", media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@app.get("/sales/national/" ,summary="Predict revenue next days ")
async def predict_sales(steps:int):
    """_summary_

    Args:
        steps:int : number of forecast to make

    Raises:
        HTTPException: _description_

    Returns:
        _type_: _description_
    """
    try:
        prediction_model = pickle.load(open('forecast_model.pkl', 'rb'))
        # prediction_model.predict
        res = list(prediction_model.forecast(steps))
        str_val = ""
        for i in res:
            str_val += " " + str(round(i,2))
        return Response(content=f"The output is for the item for next {steps} days  are: "+ str_val, media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    



if __name__ == "__main__":
    import uvicorn
    import os  # Add this import
    port = int(os.environ.get("PORT", 8000))  # Use the PORT environment variable or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)