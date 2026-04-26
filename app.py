import streamlit as st
import pandas as pd
import numpy as np
import pickle, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("🏠 House Price Predictor")
st.sidebar.markdown("---")
st.sidebar.header("Enter House Details")

area            = st.sidebar.slider("Area (sq ft)",          500,  5000, 1500)
bedrooms        = st.sidebar.slider("Bedrooms",               1,    6,    3)
bathrooms       = st.sidebar.slider("Bathrooms",              1,    4,    2)
stories         = st.sidebar.slider("Stories",                1,    4,    1)
parking         = st.sidebar.slider("Parking Spots",          0,    3,    1)
mainroad        = st.sidebar.selectbox("Main Road?",          ["Yes","No"])
guestroom       = st.sidebar.selectbox("Guest Room?",         ["Yes","No"])
basement        = st.sidebar.selectbox("Basement?",           ["Yes","No"])
hotwaterheating = st.sidebar.selectbox("Hot Water Heating?",  ["Yes","No"])
airconditioning = st.sidebar.selectbox("Air Conditioning?",   ["Yes","No"])
prefarea        = st.sidebar.selectbox("Preferred Area?",     ["Yes","No"])
furnishing      = st.sidebar.selectbox("Furnishing Status",   ["furnished","semi-furnished","unfurnished"])

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    PKL = os.path.join(os.path.dirname(__file__), "model.pkl")
    if os.path.exists(PKL):
        with open(PKL, "rb") as f:
            bundle = pickle.load(f)
        return bundle["linear_regression"], bundle["scaler"]
    # Fallback: train on the fly
    rng = np.random.default_rng(42)
    n   = 500
    area        = rng.integers(500,  5000, n)
    bedrooms    = rng.integers(1,    6,    n)
    bathrooms   = rng.integers(1,    4,    n)
    stories     = rng.integers(1,    4,    n)
    parking     = rng.integers(0,    3,    n)
    mainroad    = rng.integers(0,    2,    n)
    guestroom   = rng.integers(0,    2,    n)
    basement    = rng.integers(0,    2,    n)
    hotwaterheating = rng.integers(0, 2,  n)
    airconditioning = rng.integers(0, 2,  n)
    prefarea    = rng.integers(0,    2,    n)
    furnishing  = rng.integers(0,    3,    n)
    price = (
        area * 50 + bedrooms * 200_000 + bathrooms * 150_000 +
        stories * 100_000 + parking * 80_000 + mainroad * 200_000 +
        guestroom * 100_000 + basement * 80_000 + hotwaterheating * 60_000 +
        airconditioning * 120_000 + prefarea * 250_000 - furnishing * 50_000 +
        rng.normal(0, 200_000, n)
    ).clip(300_000, None)
    df = pd.DataFrame({
        "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "stories": stories, "parking": parking, "mainroad": mainroad,
        "guestroom": guestroom, "basement": basement,
        "hotwaterheating": hotwaterheating, "airconditioning": airconditioning,
        "prefarea": prefarea, "furnishing": furnishing, "price": price
    })
    X = df.drop("price", axis=1)
    y = np.log(df["price"])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    model  = LinearRegression().fit(scaler.fit_transform(X_train), y_train)
    return model, scaler

model, scaler = load_model()

# ── Main Page ─────────────────────────────────────────────────────────────────
st.title("🏠 House Price Prediction")
st.markdown("Using **Linear Regression** — the best performing model.")
st.markdown("---")
st.markdown("👈 Adjust the house details in the **sidebar**, then click **Predict Price**.")

if st.button("💡 Predict Price", use_container_width=True):
    fmap = {"Yes": 1, "No": 0,
            "furnished": 0, "semi-furnished": 1, "unfurnished": 2}
    inp   = np.array([[
        area, bedrooms, bathrooms, stories, parking,
        fmap[mainroad], fmap[guestroom], fmap[basement],
        fmap[hotwaterheating], fmap[airconditioning],
        fmap[prefarea], fmap[furnishing]
    ]])
    price_pred = np.exp(model.predict(scaler.transform(inp))[0])

    st.success(f"### 🏡 Predicted House Price: ₹ {price_pred:,.2f}")
    st.caption("Model: Linear Regression")

    st.markdown("#### 📋 Input Summary")
    summary = {
        "Area (sq ft)": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
        "Stories": stories, "Parking": parking, "Main Road": mainroad,
        "Guest Room": guestroom, "Basement": basement,
        "Hot Water": hotwaterheating, "AC": airconditioning,
        "Preferred Area": prefarea, "Furnishing": furnishing
    }
    st.dataframe(pd.DataFrame([summary]), use_container_width=True)
