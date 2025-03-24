from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from utils.prediction import predict
from utils.account import (
    create_account, 
    get_api_key, 
    change_password, 
    account_info, 
    create_new_api_key, 
    get_user_usage
)
from pydantic import BaseModel

#Definition of request model for prediction endpoint
class PredictionRequest(BaseModel):
    api: str
    eircode: str
    metres_squared: float
    bedrooms: int
    bathrooms: int
    ber: str
    property_type: str

#initialize FastAPI
app = FastAPI()

#root endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}

#Endpoint to return all API keys for a user
@app.get("/ApiKey")
def ApiKeyGET(username: str, password: str):
    response = get_api_key(username, password)
    return JSONResponse(content=response)

#Endpoint to return user account info
@app.get("/UserInfo")
def accountInfoGET(username: str, password: str):
    response = account_info(username, password)
    return JSONResponse(content=response)

#Endpoint to return user usage
@app.get("/UserUsage")
def UserUsageGET(username: str, password: str):
    response = get_user_usage(username, password)
    return JSONResponse(content=response)

#Endpoint to make a prediction given specified parameters
@app.post("/Predict")
def predictPOST(request: PredictionRequest):
    response = predict(request)
    return JSONResponse(content=response)

#Endpoint to create a new account
@app.post("/Account")
def createAccountPOST(username: str, password: str):
    response = create_account(username, password)
    return JSONResponse(content=response)

#Endpoint to change a user's password
@app.post("/Password")
def changePasswordPOST(username: str, old_password: str, new_password: str):
    response = change_password(username, old_password, new_password)
    return JSONResponse(content=response)

#Endpoint to create a new API key for a user
@app.post("/ApiKey")
def NewApiKeyPOST(username: str, password: str):
    response = create_new_api_key(username, password)
    return JSONResponse(content=response)