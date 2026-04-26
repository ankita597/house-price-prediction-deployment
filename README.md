# 🏠 House Price Prediction — Deployment Guide

## Project Overview
Predicts house prices using **Linear Regression** and **KNN Regressor** with an interactive Streamlit web app.

## Features
- 📊 Dataset overview with histograms & correlation heatmap
- 📈 Model evaluation — MAE, RMSE, R² for both models
- 🔮 Live prediction with sidebar sliders

## Local Setup

```bash
# 1. Clone / copy project files
cd house_price_app

# 2. (Optional) create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

App opens at **http://localhost:8501**

## ☁️ Deploy on Streamlit Community Cloud (FREE)

1. Push this folder to a **GitHub repository**
2. Go to https://share.streamlit.io → **New app**
3. Select your repo → branch → set **Main file path** = `app.py`
4. Click **Deploy** — live URL generated in ~2 min ✅

## ☁️ Deploy on Render (FREE)

1. Push to GitHub
2. Go to https://render.com → **New Web Service**
3. Connect repo, set:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy ✅

## Models Used
| Model | Type |
|---|---|
| Linear Regression | Parametric, fast baseline |
| KNN Regressor | Non-parametric, k=5 neighbors |

## Tech Stack
`Python` · `Streamlit` · `scikit-learn` · `pandas` · `numpy` · `matplotlib` · `seaborn`
