
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained ANN model
model = load_model("model.h5")

# Load fitted scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    probability = None
    error = None

    if request.method == "POST":
        try:
            pclass = int(request.form["Pclass"])
            sex = int(request.form["Sex"])
            sibsp = int(request.form["SibSp"])
            parch = int(request.form["Parch"])
            fare = float(request.form["Fare"])
            embarked = request.form["Embarked"]

            # One-hot encode Embarked using the same columns
            embarked_cherbourg = 1 if embarked == "C" else 0
            embarked_queenstown = 1 if embarked == "Q" else 0
            embarked_southampton = 1 if embarked == "S" else 0

            # Keep EXACT same feature order as training
            columns = [
                "Pclass", "Sex", "SibSp", "Parch", "Fare",
                "Embarked_Cherbourg",
                "Embarked_Queenstown",
                "Embarked_Southampton"
            ]

            user = pd.DataFrame([[
                pclass, sex, sibsp, parch, fare,
                embarked_cherbourg,
                embarked_queenstown,
                embarked_southampton
            ]], columns=columns)

            # Scale only the same columns that were scaled in training
            user[["Pclass", "SibSp", "Parch", "Fare"]] = (
                scaler.transform(
                    user[["Pclass", "SibSp", "Parch", "Fare"]]
                )
            )

            result = model.predict(user, verbose=0)
            probability = float(result[0][0]) * 100

            if result[0][0] > 0.5:
                prediction = "Survived"
            else:
                prediction = "Not Survived"

        except Exception as e:
            error = f"Prediction error: {e}"

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability,
        error=error
    )


if __name__ == "__main__":
    app.run(debug=True)