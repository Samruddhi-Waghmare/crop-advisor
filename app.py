import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Krishi Mitra", page_icon="🌾", layout="wide")

# ---------- Load models (cached so they load only once) ----------
@st.cache_resource
def load_all():
    rec = joblib.load("models/recommend_xgb.pkl")
    le = joblib.load("models/label_encoder.pkl")
    yld = joblib.load("models/yield_xgb.pkl")
    meta = joblib.load("models/meta.pkl")
    return rec, le, yld, meta

rec_model, le, yield_model, meta = load_all()

# Recommendation dataset names -> production dataset names
CROP_MAP = {
    "rice": "Rice", "maize": "Maize", "chickpea": "Gram",
    "pigeonpeas": "Arhar/Tur", "mothbeans": "Moth",
    "mungbean": "Moong(Green Gram)", "blackgram": "Urad",
    "lentil": "Masoor", "banana": "Banana", "mango": "Mango",
    "grapes": "Grapes", "watermelon": "Water Melon", "apple": "Apple",
    "orange": "Orange", "papaya": "Papaya",
    "cotton": "Cotton(lint)", "jute": "Jute", "coffee": "Coffee",
}

# ---------- English / Hindi text ----------
T = {
    "en": {
        "title": "🌾 Krishi Mitra: Smart Crop Advisor",
        "sub": "Enter your soil and weather details to get crop suggestions and expected yield.",
        "state": "State", "season": "Season", "area": "Land area (hectares)",
        "n": "Nitrogen (N)", "p": "Phosphorus (P)", "k": "Potassium (K)",
        "temp": "Temperature (°C)", "hum": "Humidity (%)", "ph": "Soil pH",
        "rain": "Rainfall (mm)", "btn": "Get Recommendation",
        "top": "Top 3 recommended crops", "conf": "Confidence",
        "yield": "Expected yield (tonnes/hectare)",
        "total": "Estimated total production (tonnes)",
        "nodata": "Yield data not available",
    },
    "hi": {
        "title": "🌾 कृषि मित्र: स्मार्ट फसल सलाहकार",
        "sub": "फसल सुझाव और अनुमानित उपज पाने के लिए अपनी मिट्टी और मौसम की जानकारी दें।",
        "state": "राज्य", "season": "मौसम", "area": "ज़मीन का क्षेत्रफल (हेक्टेयर)",
        "n": "नाइट्रोजन (N)", "p": "फॉस्फोरस (P)", "k": "पोटैशियम (K)",
        "temp": "तापमान (°C)", "hum": "नमी (%)", "ph": "मिट्टी का pH",
        "rain": "वर्षा (मिमी)", "btn": "सलाह पाएं",
        "top": "सबसे अच्छी 3 फसलें", "conf": "विश्वास",
        "yield": "अनुमानित उपज (टन/हेक्टेयर)",
        "total": "अनुमानित कुल उत्पादन (टन)",
        "nodata": "उपज का डेटा उपलब्ध नहीं",
    },
}

lang = st.sidebar.radio("भाषा / Language", ["English", "हिन्दी"])
t = T["en"] if lang == "English" else T["hi"]

st.title(t["title"])
st.write(t["sub"])

# ---------- Inputs ----------
c1, c2, c3 = st.columns(3)
with c1:
    state = st.selectbox(t["state"], meta["states"])
    season = st.selectbox(t["season"], meta["seasons"])
    area = st.number_input(t["area"], min_value=0.1, value=2.0, step=0.5)
with c2:
    N = st.slider(t["n"], 0, 140, 60)
    P = st.slider(t["p"], 5, 145, 50)
    K = st.slider(t["k"], 5, 205, 50)
with c3:
    temp = st.slider(t["temp"], 8.0, 44.0, 25.0)
    hum = st.slider(t["hum"], 14.0, 100.0, 70.0)
    ph = st.slider(t["ph"], 3.5, 9.9, 6.5)
    rain = st.slider(t["rain"], 20.0, 300.0, 100.0)

# ---------- Predict ----------
if st.button(t["btn"], type="primary"):
    X = pd.DataFrame(
        [[N, P, K, temp, hum, ph, rain]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )
    probs = rec_model.predict_proba(X)[0]
    top3 = probs.argsort()[::-1][:3]

    st.subheader(t["top"])
    cols = st.columns(3)
    for col, idx in zip(cols, top3):
        crop = le.classes_[idx]
        with col:
            st.markdown(f"### {crop.title()}")
            st.metric(t["conf"], f"{probs[idx] * 100:.1f}%")

            prod_name = CROP_MAP.get(crop)
            if prod_name in meta["crops"]:
                row = pd.DataFrame([{
                    "State_Name": state, "Season": season, "Crop": prod_name,
                    "Crop_Year": meta["latest_year"], "Area": area,
                }])
                y = float(yield_model.predict(row)[0])
                st.metric(t["yield"], f"{y:.2f}")
                st.caption(f"{t['total']}: {y * area:.1f}")
            else:
                st.caption(t["nodata"])







import plotly.express as px

# ---------- Crop explorer ----------
@st.cache_data
def load_yield_table():
    raw = pd.read_csv("data/crop_production.csv").dropna(subset=["Production"])
    for col in ["State_Name", "Season", "Crop"]:
        raw[col] = raw[col].str.strip()
    raw = raw[raw["Area"] > 0]
    raw["Yield"] = raw["Production"] / raw["Area"]
    return raw[raw["Yield"] < raw["Yield"].quantile(0.99)]

st.divider()
st.subheader("📊 Crop explorer" if lang == "English" else "📊 फसल एक्सप्लोरर")
table = load_yield_table()
crop_choice = st.selectbox("Crop", meta["crops"])
avg = (table[table["Crop"] == crop_choice]
       .groupby("State_Name")["Yield"].mean().reset_index()
       .sort_values("Yield"))
fig = px.bar(avg, x="Yield", y="State_Name", orientation="h",
             title=f"Average yield of {crop_choice} by state (tonnes/hectare)")
st.plotly_chart(fig, use_container_width=True)