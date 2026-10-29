
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Steel Sheet Defect Predictor",
    page_icon="🏭",
    layout="centered"
)

st.title("🏭 Steel Sheet Defect Prediction")
st.write("Enter manufacturing parameters to predict the product condition.")

MODEL_PATH = Path(__file__).parent / "manufacturing_defect_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
    st.success("Machine-learning model loaded successfully!")
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

st.warning(
    "Academic prototype: the model uses general manufacturing data, "
    "not verified steel-sheet production data."
)

st.header("Manufacturing parameters")

production_line = st.selectbox(
    "Production line", ["LineA", "LineB", "LineC"]
)
temperature = st.number_input("Temperature", value=75.0)
pressure = st.number_input("Pressure", value=90.0)
assembly_time = st.number_input("Processing time", value=180.0)
sensor1 = st.number_input("Sensor 1 reading", value=0.9)
sensor2 = st.number_input("Sensor 2 reading", value=1.0)
sensor3 = st.number_input("Sensor 3 reading", value=1.0)
operator_experience = st.number_input("Operator experience", value=5.0)

if st.button("Predict product condition"):
    input_data = pd.DataFrame([{
        "ProductionLine": production_line,
        "Temperature": temperature,
        "Pressure": pressure,
        "AssemblyTime": assembly_time,
        "Sensor1Reading": sensor1,
        "Sensor2Reading": sensor2,
        "Sensor3Reading": sensor3,
        "OperatorExperience": operator_experience
    }])

    try:
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)
        defect_probability = probabilities[classes.index(1)]

        if int(prediction) == 1:
            st.error("🔴 DEFECTIVE — predicted defect class")
        else:
            st.success("🟢 NORMAL — predicted non-defect class")

        st.metric(
            "Estimated defect probability",
            f"{defect_probability * 100:.2f}%"
        )
        st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Prototype only. Validate against real steel-sheet data.")
