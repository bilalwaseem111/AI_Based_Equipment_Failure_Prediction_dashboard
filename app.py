import streamlit as st
import joblib
import plotly.express as px
from PIL import Image
import os

# Page Config
st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# CSS Styling
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
   background-color: #353535;  /* Main page light grey */
}
[data-testid="stSidebar"] {
    background-color: #706f6f; /* Sidebar light grey */
}
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4 {
    color: #f6ff00; /* Sidebar titles color */
}
.title {
    font-size: 62px;
    text-align: center;
    color: #2bff00;
    font-weight: bold;
    margin-bottom: 25px;
    -webkit-text-stroke: 2px rgb(226, 11, 223);
    text-shadow: 0 0 10px red, 0 0 20px red; /* Initial red glow */
    animation: glowAnimation 1s infinite alternate;
}

@keyframes glowAnimation {
    0% {
        text-shadow: 0 0 10px red, 0 0 20px red, 0 0 30px red;
    }
    100% {
          text-shadow: 0 0 10px rgb(243, 16, 114), 0 0 20px rgb(211, 7, 113), 0 0 30px rgb(114, 49, 81);
    }
}





.machine-img {
    display: block;
    margin: auto;
    width: 380px;
    border-radius: 15px;
    box-shadow: 0 4px 15px rgba(255,255,255,0.2);
}
.stButton button {
    background: linear-gradient(90deg, #FF512F, #DD2476);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 12px;
    transition: 0.3s;
}
.stButton button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #DD2476, #FF512F);
}
.card {
    background: #222;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(255,255,255,0.15);
    margin-bottom: 20px;
}
.footer {
    text-align: center;
    margin-top: 20px;
    font-size: 18px;
    color: white;
    font-weight: bold;
}
            /* Sidebar headers and labels in white */
[data-testid="stSidebar"] h1, 
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] h3, 
[data-testid="stSidebar"] h4, 
[data-testid="stSidebar"] label {
    color: white !important;
}

</style>
""", unsafe_allow_html=True)
# Header
st.markdown("<div class='title'>AI Based Equipment Failure Prediction Dashboard</div>", unsafe_allow_html=True)

# Sidebar - Machine selection
st.sidebar.header("Select Machine Type")
machine_types = ["Pump", "Motor", "Compressor", "Conveyor", "Fan", "Gas-Turbine", "Generator", "Gearbox", "Hydraulic Press", "Drill"]
machine_choice = st.sidebar.selectbox("Choose Machine", machine_types)

# Sidebar Inputs
st.sidebar.subheader(f"Parameters for {machine_choice}")
air_temp = st.sidebar.slider("Air Temperature (K)", 290, 320, 300)
process_temp = st.sidebar.slider("Process Temperature (K)", 300, 350, 325)
rot_speed = st.sidebar.slider("Rotational Speed (rpm)", 1200, 3000, 1500)
torque = st.sidebar.slider("Torque (Nm)", 0, 100, 50)
tool_wear = st.sidebar.slider("Tool Wear (min)", 0, 250, 100)

# Machine Overview
machine_details = {
    "Pump": "Used to move fluids. Failure often occurs due to overheating or excessive wear.",
    "Motor": "Converts electrical energy to mechanical. Failure risk rises with high torque & speed.",
    "Compressor": "Increases air/gas pressure. Fails with high process temperature & torque.",
    "Conveyor": "Moves materials. Risk increases with mechanical wear.",
    "Fan": "Maintains airflow. Overheating can cause failure.",
    "Gas-Turbine": "Generates power. High process heat causes critical failures.",
    "Generator": "Produces electricity. High load and wear reduce lifespan.",
    "Gearbox": "Transfers power. Torque spikes lead to failure.",
    "Hydraulic Press": "Applies force for molding. Overpressure is dangerous.",
    "Drill": "Cuts material. Worn tools & overheating cause damage."
}

# Always show machine info and image
st.markdown(f"""
<div class='card' style='padding:15px; border-radius:10px; background-color:#060606; text-align:center;'>
    <h3 style='color:#fcfcfb;'>{machine_choice} Overview</h3>
    <p style='color:#f2ff00; font-size:18px;'>{machine_details[machine_choice]}</p>
</div>
""", unsafe_allow_html=True)

# Display Machine Image
image_path = f"images/{machine_choice.lower().replace(' ', '_')}.jpg"
if os.path.exists(image_path):
    img = Image.open(image_path)
    st.image(img, caption=f"{machine_choice} Illustration", use_container_width=False)

# Prediction Logic
predict_btn = st.sidebar.button("Predict Machine Health")

if predict_btn:
    input_data = [[air_temp, process_temp, rot_speed, torque, tool_wear]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    # Failure probability
    try:
        failure_prob = model.predict_proba(input_scaled)[0][1] * 100
    except:
        failure_prob = 50

    # Status & Suggestion
    if prediction == 0:
        status = "Healthy"
        color = "white"
        suggestion = "No immediate action required. Continue routine checks."
    else:
        status = "Failure Risk"
        color = "white"
        suggestion = "Schedule maintenance within the next 48 hours."

    # Estimated failure time
    est_failure_time = round((300 - tool_wear) / (torque / 10 + 1), 1)

    # Results Section
    st.markdown(f"""
        <div style=' background-color:#060606; padding: 25px; border-radius: 18px; text-align:center; 
                    box-shadow: 0 4px 15px rgba(255,255,255,0.15); margin-top:20px;'>
            <h3 style='color:{color}; font-size:30px; margin-bottom:15px;'>Status: {status}</h3>
            <div style='background: rgba(255,255,255,0.08); padding:15px; border-radius:10px;'>
                <h4 style='color:#f2ff00; font-size:22px; margin:10px 0;'>Failure Probability: {failure_prob:.2f}%</h4>
                <h4 style='color:#f2ff00; font-size:22px; margin:10px 0;'>Estimated Failure in: {est_failure_time} hrs</h4>
                <h4 style='color:#f2ff00; font-size:22px; margin:10px 0;'>Suggested Action: {suggestion}</h4>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Big Heading for 3D Visualization
    st.markdown(f"<h2 style='text-align:center; color:black; font-size:36px;'>{machine_choice} Health Visualization</h2>", unsafe_allow_html=True)

    # 3D Health Visualization
    fig = px.scatter_3d(
        x=[air_temp], y=[process_temp], z=[rot_speed],
        color_discrete_map={"Healthy": "white", "Failure Risk": "white"},
        size=[torque + 10],
        title=""
    )
    fig.update_layout(scene=dict(
        xaxis_title="Air Temp",
        yaxis_title="Process Temp",
        zaxis_title="Rot Speed"
    ))
    st.plotly_chart(fig, use_container_width=True)

    # Big Heading for Bar Chart
    st.markdown("<h2 style='text-align:center; color:#000000; font-size:36px;'>Parameter Impact Overview</h2>", unsafe_allow_html=True)

    # Feature Impact Bar Chart
    features = ["Air Temp", "Process Temp", "Rot Speed", "Torque", "Tool Wear"]
    values = [air_temp, process_temp, rot_speed, torque, tool_wear]

    bar_fig = px.bar(
        x=features,
        y=values,
        text=values,
        title="",
        color=values,
        color_continuous_scale="Viridis"
    )
    bar_fig.update_traces(textposition='outside')
    st.plotly_chart(bar_fig, use_container_width=True)

# Footer
st.markdown("<div style='text-align:center; color:white; margin-top:20px;'>Made by Bilal Waseem</div>", unsafe_allow_html=True)
