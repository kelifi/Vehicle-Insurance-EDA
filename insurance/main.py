import pickle

import pandas as pd
import uvicorn
from fastapi import FastAPI

app = FastAPI()

# Load the Random Forest model
filename = 'rf_model.sav'
rf_load = pickle.load(open(filename, 'rb'))


@app.post("/predict/")
def predict(features: dict):
    print('hello')
    """
    Endpoint to make predictions using the Random Forest model.
    :param features: Input features as a dictionary.
    :return: Predicted class or value.
    """
    input_data = pd.DataFrame(features)
    try:

        prediction = rf_load.predict(input_data)

        return {"prediction": prediction[0]}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
