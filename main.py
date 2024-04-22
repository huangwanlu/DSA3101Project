#pip install fastapi
#pip install "uvicorn[standard]"
#pip install python-multipart

#IMPT! Before starting in development* server, must run the uvicorn server using "uvicorn main:app --reload" in terminal

from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from io import BytesIO

# load model from pickle file
with open('./back/final_model.pkl', 'rb') as file:  
    model = pickle.load(file)
    
#load country encoder
with open('./back/country_encoder.pkl', 'rb') as file:
    country_encoder = pickle.load(file)

#load gender encoder
with open('./back/gender_encoder.pkl', 'rb') as file:
    gender_encoder = pickle.load(file)

origins = [
    "http://localhost:3000",
    "https://localhost:3000",
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins, #or can specify wildcard ["*"] which allows from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#takes in a param called 'file' (rmb to match with what u call in FormData) with type UploadFile
@app.post('/uploadfile')
async def upload_file(file_upload: UploadFile): 
    if not file_upload: 
        return {"message": "No file sent"}
    else:    
        #reading in data
        data = await file_upload.read()
        df = pd.read_csv(BytesIO(data))
        
        #transforming dataframe for prediction
        ids = df['customer_id']
        country = df['country']
        gender = df['gender']
        results = df.drop('customer_id', axis=1)
        
        #encoding 'country' and 'gender' columns
        results['country'] = country_encoder.transform(results['country']) 
        results['gender'] = gender_encoder.transform(results['gender']) 
        
        #predicting churn
        results['churn'] = model.predict(results)
        
        #cleaning up dataframe for frontend display
        final = pd.concat([pd.Series(ids, index=results.index, name='customer_id'), results], axis=1)
        final['country'] = country
        final['gender'] = gender
        
        return(final.to_dict(orient='records'))
    
        # #exporting as JSON file
        # final.to_json('./uploads/data.json', orient='records')
        
        # return {"filenames": file_upload.filename}

# app.mount("/uploads", StaticFiles(directory="./uploads"), name="uploads")
# @app.get("/get_data")
# async def get_data():
#     return {"file_path": "/uploads/data.json"}