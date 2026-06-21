"""
Smart Crop Recommendation System — Dashboard & Frontend
-----------------------------------------------------------
A single-file Streamlit app that combines:
  1. A KPI dashboard (model performance + dataset insights)
  2. An interactive UI where a user enters soil/climate readings
     and gets a recommended crop in real time.

Run:
    streamlit run app.py
"""

import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from google import genai

# ---------------------------------------------------------------- CONFIG --
st.set_page_config(
    page_title="Smart Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
)

PRIMARY_GREEN = "#1B4332"
SOIL_BROWN = "#6B4226"
WHEAT_GOLD = "#D4A24C"
BG = "#F1F4EC"

st.markdown(f"""
<style>
.stApp {{ background-color: {BG}; }}
h1, h2, h3 {{ font-family: 'Trebuchet MS', sans-serif; color: {PRIMARY_GREEN}; }}
.kpi-card {{
    background: white; border-radius: 14px; padding: 18px 20px;
    border-left: 6px solid {WHEAT_GOLD};
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.kpi-value {{ font-size: 30px; font-weight: 700; color: {PRIMARY_GREEN}; font-family: monospace; }}
.kpi-label {{ font-size: 13px; color: #555; text-transform: uppercase; letter-spacing: 0.5px; }}
.horizon-bar {{
    height: 6px; border-radius: 4px; margin: 6px 0 22px 0;
    background: linear-gradient(90deg, {WHEAT_GOLD}, {PRIMARY_GREEN}, {SOIL_BROWN});
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- LOAD ----
@st.cache_resource
def load_model():
    model = joblib.load("crop_model.pkl")
    features = joblib.load("features.pkl")
    return model, features

@st.cache_data
def load_dashboard_data():
    with open("dashboard_data.json") as f:
        return json.load(f)

model, FEATURES = load_model()
data = load_dashboard_data()

# ---------------------------------------------------------------- SIDEBAR --
with st.sidebar:
    st.header("🤖 Crop Advisor Chatbot")
    api_key = st.text_input(
        "Google API key",
        type="password",
        help="Get one free at console.anthropic.com/account/keys. "
             "Used only for this session — never stored or sent anywhere else."
    )
    st.caption("Needed only for the chatbot below the recommendation. "
               "The crop prediction itself works without it.")

# ---------------------------------------------------------------- HEADER --
st.title("🌱 Smart Crop Recommendation System")
st.caption("AI-powered crop recommendation based on soil nutrients (N, P, K) and climate conditions.")
st.markdown('<div class="horizon-bar"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- KPI ROW --
kpi_cols = st.columns(5)
kpis = [
    ("Model Accuracy", f"{data['accuracy']*100:.1f}%"),
    ("F1 Score (macro)", f"{data['f1_macro']*100:.1f}%"),
    ("Crops Covered", f"{data['n_classes']}"),
    ("Training Samples", f"{data['train_size']:,}"),
    ("Test Samples", f"{data['test_size']:,}"),
]
for col, (label, value) in zip(kpi_cols, kpis):
    col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------------- BODY ----
left, right = st.columns([1.1, 1])

with left:
    st.subheader("📊 Model Insights")

    fi = pd.DataFrame(
        sorted(data["feature_importance"].items(), key=lambda x: x[1]),
        columns=["Feature", "Importance"]
    )
    fig_fi = px.bar(
        fi, x="Importance", y="Feature", orientation="h",
        color_discrete_sequence=[PRIMARY_GREEN],
        title="Which factors matter most for the recommendation?"
    )
    fig_fi.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor="white")
    st.plotly_chart(fig_fi, use_container_width=True)

    f1_df = pd.DataFrame(
        sorted(data["per_class_f1"].items(), key=lambda x: x[1]),
        columns=["Crop", "F1 Score"]
    )
    fig_f1 = px.bar(
        f1_df, x="F1 Score", y="Crop", orientation="h",
        color="F1 Score", color_continuous_scale=["#C0392B", WHEAT_GOLD, PRIMARY_GREEN],
        title="Per-crop prediction reliability (F1 score)"
    )
    fig_f1.update_layout(height=520, margin=dict(l=0, r=0, t=40, b=0), plot_bgcolor="white")
    st.plotly_chart(fig_f1, use_container_width=True)

    with st.expander("ℹ️ Why isn't accuracy higher than ~74%?"):
        st.write(
            "A handful of crops (apple, orange, mango, grapes, pomegranate) share "
            "almost identical soil nutrient profiles and only differ subtly in "
            "temperature/rainfall. The model genuinely confuses these — which is "
            "expected and worth discussing honestly in your project report, rather "
            "than hiding it. Adding features like altitude, season, or soil moisture "
            "would likely resolve this overlap."
        )

with right:
    st.subheader("🔍 Get a Crop Recommendation")
    st.write("Enter the soil test and weather readings below:")

    c1, c2 = st.columns(2)
    with c1:
        N = st.slider("Nitrogen — N (kg/ha)", 0, 160, 50)
        P = st.slider("Phosphorus — P (kg/ha)", 0, 160, 50)
        K = st.slider("Potassium — K (kg/ha)", 0, 210, 50)
        ph = st.slider("Soil pH", 3.0, 10.0, 6.5, step=0.1)
    with c2:
        temperature = st.slider("Temperature (°C)", 0.0, 45.0, 25.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 300.0, 100.0)

    if st.button("🌾 Recommend Crop", use_container_width=True):
        input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=FEATURES)
        probs = model.predict_proba(input_df)[0]
        top_idx = probs.argsort()[::-1][:3]
        top_crops = model.classes_[top_idx]
        top_probs = probs[top_idx]

        # Save to session_state so it survives the chatbot's reruns below
        st.session_state["last_prediction"] = {
            "top_crops": top_crops.tolist(),
            "top_probs": top_probs.tolist(),
            "inputs": {"N": N, "P": P, "K": K, "temperature": temperature,
                       "humidity": humidity, "ph": ph, "rainfall": rainfall},
        }
        st.session_state["chat_history"] = []  # reset chat for the new crop

    if "last_prediction" in st.session_state:
        pred = st.session_state["last_prediction"]
        top_crops = pred["top_crops"]
        top_probs = pred["top_probs"]

        st.success(f"**Recommended crop: {top_crops[0].title()}**")

        fig_conf = go.Figure(go.Bar(
            x=[p * 100 for p in top_probs],
            y=[c.title() for c in top_crops],
            orientation="h",
            marker_color=[PRIMARY_GREEN, WHEAT_GOLD, SOIL_BROWN]
        ))
        fig_conf.update_layout(
            title="Top 3 candidate crops",
            xaxis_title="Confidence (%)",
            height=260, margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_conf, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------- CHATBOT --
if "last_prediction" in st.session_state:
    crop_name = st.session_state["last_prediction"]["top_crops"][0]
    inputs = st.session_state["last_prediction"]["inputs"]

    st.subheader(f"💬 Ask the Crop Advisor about growing {crop_name.title()}")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if not st.session_state["chat_history"]:
        st.caption(f"Try asking: \"What's the full growing process for {crop_name}?\", "
                    "\"What fertilizer schedule should I follow?\", or \"What pests should I watch for?\"")

    user_msg = st.chat_input(f"Ask anything about growing {crop_name}...")

    if user_msg:
        if not api_key:
            st.error("Add your Google API key in the sidebar first.")
        else:
            st.session_state["chat_history"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.write(user_msg)

            system_prompt = (
                f"You are an agronomy assistant inside a crop recommendation app. "
                f"The model just recommended growing {crop_name} based on these readings: "
                f"N={inputs['N']}, P={inputs['P']}, K={inputs['K']}, "
                f"temperature={inputs['temperature']}°C, humidity={inputs['humidity']}%, "
                f"pH={inputs['ph']}, rainfall={inputs['rainfall']}mm. "
                f"Answer the farmer's questions about growing {crop_name}: sowing season, "
                f"soil preparation, irrigation, fertilizer schedule, common pests/diseases, "
                f"and harvest timing. Keep answers practical and concise (under 200 words "
                f"unless asked for more detail)."
            )

            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(model="gemini-2.5-flash",contents=f"{system_prompt}\n\nUser Question: {user_msg}")
                reply = response.text
            except Exception as e:
                reply = f"Couldn't reach the chatbot — check that your API key is correct. ({e})"

            st.session_state["chat_history"].append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)
else:
    st.info("Get a crop recommendation above to unlock the chatbot.")

st.markdown("---")
st.caption("Smart Crop Recommendation System · Random Forest classifier trained on 15,000 soil/climate records across 23 crops · AI Internship Project")
