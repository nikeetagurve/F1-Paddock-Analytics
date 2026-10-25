# F1 Paddock Analytics 🏁

A Formula 1-inspired Streamlit dashboard exploring the final 2024 Formula 1 Drivers' Championship standings through interactive visualizations, statistical summaries, regression models, and classification models.

## Features
- Race Control: championship snapshot and team points visualizations
- Driver Grid: searchable final standings and driver performance map
- Data Garage: dataset inspection, descriptive statistics, data types, interactive charts, CSV export
- Points Lab: comparison of five regression models and a live points estimate
- Top 10 Predictor: comparison of five classifiers, user-controlled predictions, probability display, and confusion matrix
- Methodology: workflow and official sources

## Run locally
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Data sources
- [Formula 1 — 2024 Drivers' Standings](https://www.formula1.com/en/results/2024/drivers)
- [FIA — 2024 Classifications](https://www.fia.com/events/fia-formula-one-world-championship/season-2024/2024-classifications-0)

## Project details
The dashboard is designed as an educational SI Lab project and presents results for the completed 2024 season. The dataset is embedded locally, so the app does not need to scrape a website at runtime.


## 🚀 Live Demo

[Open F1 Paddock Analytics](https://f1-paddock-analytics-g753ffmnqyedgxv7zx2f6e.streamlit.app)

Explore 2024 Formula 1 driver standings, interactive visualizations, regression models, and Top 10 classification predictions.
