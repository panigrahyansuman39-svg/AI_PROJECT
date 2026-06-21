# Smart Crop Recommendation System
AI Internship Project — End-to-End Workflow

## 1. What this project does
Given soil nutrient readings (Nitrogen, Phosphorus, Potassium, pH) and climate
conditions (temperature, humidity, rainfall), the model recommends the most
suitable crop to grow. Trained on 15,000 records covering 23 crops.

## 2. Files in this project
| File | Purpose |
|---|---|
| `Cleaned_Dataset.xlsx` | Your dataset (15,000 rows, 23 crops) |
| `train_model.py` | Trains the Random Forest model, saves it, exports metrics |
| `crop_model.pkl` | The trained model (already generated for you) |
| `features.pkl` | List of feature columns the model expects |
| `dashboard_data.json` | Metrics/charts data used by the dashboard |
| `app.py` | The dashboard + interactive frontend (Streamlit) |
| `requirements.txt` | Python packages needed |

## 3. How to run it locally
```bash
pip install -r requirements.txt
python train_model.py      # only needed if you change the dataset/model
streamlit run app.py
```
This opens a browser window with the full dashboard + prediction UI — nothing
else to configure.

## 4. How to submit / deploy it (free, no server needed)
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io → "New app" → connect your repo → select `app.py`.
3. You get a public URL in ~2 minutes. Put that link in your internship report.

## 4b. Chatbot: getting an Anthropic API key
The dashboard includes an AI chatbot that explains the full growing process for
whichever crop gets recommended (sowing season, irrigation, fertilizer, pests,
harvest). It uses the Claude API, so you need a key:
1. Go to https://console.anthropic.com/account/keys and sign up.
2. Create a new API key (new accounts get free starter credit — plenty for a demo).
3. Paste the key into the sidebar of the running app. It's only kept in that
   session — never written to disk, never committed to GitHub.
4. If deploying on Streamlit Cloud and you want the key pre-filled instead of
   typed each time, add it under your app's **Settings → Secrets** as
   `ANTHROPIC_API_KEY = "sk-ant-..."` — never put a real key directly in `app.py`
   or commit it to your repo.

## 5. The workflow you followed (for your report/viva)
1. **Problem definition** — predict the best crop from soil + climate data.
2. **EDA** — checked class balance (~650 samples/crop, no missing values),
   correlations, and per-crop feature averages.
3. **Preprocessing** — selected 7 numeric features; no scaling needed for
   tree-based models; 80/20 stratified train/test split.
4. **Model selection** — compared Logistic Regression, Decision Tree, KNN,
   Nearest Centroid, and Random Forest. Random Forest won (best accuracy + F1).
5. **Training** — Random Forest, 300 trees, `min_samples_leaf=2`.
6. **Evaluation** — accuracy, macro F1/precision/recall, confusion matrix,
   per-class F1 (see `dashboard_data.json`).
7. **Explainability** — feature importance ranking (rainfall, N, temperature
   are the top drivers).
8. **Dashboard** — KPI cards + charts built with Streamlit + Plotly.
9. **Frontend** — slider-based input form, real-time prediction with
   confidence scores for the top 3 candidate crops.
10. **Deployment** — Streamlit Community Cloud (free, one click from GitHub).

## 6. Model performance (honest numbers — explain these, don't hide them)
- Accuracy: **74.0%**, F1 (macro): **74.1%**
- The model is excellent (F1 > 0.85) at telling apart crops with distinct
  profiles (wheat, papaya, cotton, coconut).
- It struggles (F1 ~0.36–0.46) with a cluster of fruits — apple, orange,
  mango, grapes, pomegranate — because they have nearly identical N/P/K and
  only subtly different temperature/rainfall. This is a genuine limitation
  of the dataset's feature set, not a bug. Mentioning this in your report
  shows you understand the model rather than just reporting a number.

## 7. Possible improvements to mention in your report
- Add features like altitude, soil moisture, or season to separate the
  confused fruit cluster.
- Try class-specific thresholds or a hierarchical classifier (first predict
  "crop family", then the specific crop).
- Collect more real-world samples for the low-F1 crops.
