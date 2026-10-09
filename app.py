import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Prediction",
    page_icon="🚢",
    layout="centered"
)

# Load trained ANN model and fitted scaler
@st.cache_resource
def load_artifacts():
    model = load_model("model.h5")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


try:
    model, scaler = load_artifacts()
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()


# Application title
st.title("🚢 Titanic Survival Prediction")
st.write(
    "Enter passenger details to predict whether "
    "the passenger survived the Titanic disaster."
)

st.divider()

# Input form
with st.form("prediction_form"):

    st.subheader("Passenger Information")

    pclass = st.selectbox(
        "Passenger Class (Pclass)",
        options=[1, 2, 3],
        index=2
    )

    sex_option = st.selectbox(
        "Gender",
        options=["Male", "Female"]
    )

    # Same encoding as training
    sex = 1 if sex_option == "Male" else 0

    sibsp = st.number_input(
        "Number of Siblings / Spouses (SibSp)",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    parch = st.number_input(
        "Number of Parents / Children (Parch)",
        min_value=0,
        max_value=10,
        value=0,
        step=1
    )

    fare = st.number_input(
        "Ticket Fare",
        min_value=0.0,
        value=32.0,
        step=1.0
    )

    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {
            "S": "Southampton (S)",
            "C": "Cherbourg (C)",
            "Q": "Queenstown (Q)"
        }[x]
    )

    submitted = st.form_submit_button(
        "Predict Survival",
        use_container_width=True
    )


# Prediction
if submitted:
    try:
        # One-hot encoding: keep EXACT training feature order
        embarked_cherbourg = 1 if embarked == "C" else 0
        embarked_queenstown = 1 if embarked == "Q" else 0
        embarked_southampton = 1 if embarked == "S" else 0

        columns = [
            "Pclass",
            "Sex",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked_Cherbourg",
            "Embarked_Queenstown",
            "Embarked_Southampton"
        ]

        user = pd.DataFrame(
            [[
                pclass,
                sex,
                sibsp,
                parch,
                fare,
                embarked_cherbourg,
                embarked_queenstown,
                embarked_southampton
            ]],
            columns=columns
        )

        # Scale only the features scaled during training
        scaled_columns = ["Pclass", "SibSp", "Parch", "Fare"]

        user[scaled_columns] = scaler.transform(
            user[scaled_columns]
        )

        # ANN prediction
        result = model.predict(user, verbose=0)
        survival_probability = float(result[0][0])

        probability_percent = survival_probability * 100

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            "Predicted Survival Probability",
            f"{probability_percent:.2f}%"
        )

        st.progress(
            max(0.0, min(1.0, survival_probability))
        )

        if survival_probability > 0.5:
            st.success("✅ Prediction: Survived")
        else:
            st.error("❌ Prediction: Not Survived")

        st.caption(
            "This is an ANN model prediction, not a guarantee "
            "of the actual outcome."
        )

        with st.expander("View model input"):
            st.dataframe(user, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
