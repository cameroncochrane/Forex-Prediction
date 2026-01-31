# Forex Prediction — Streamlit App

A Streamlit web application for exploring and running machine learning and deep learning models to predict foreign exchange (forex) prices. This repository contains model code, data utilities, example notebooks, and a small Streamlit frontend (`app.py`) that lets you load models and visualize forecasts.

**Highlights**
- Interactive Streamlit UI for running predictions and viewing results.
- Multiple modeling approaches included: ARIMA, XGBoost, and RNN-based models.
- Jupyter notebooks demonstrating analysis and experiments.

---

**Quick Links**
- App entrypoint: `app.py`
- Notebooks: `1_analysis_arima.ipynb`, `2_analysis_gradient.ipynb`, `3_analysis_rnn.ipynb`
- Model code: `arima_functions.py`, `xgboost_functions.py`, `rnn_functions.py`
- Data helpers: `data_functions.py`
- Dockerfile: `Dockerfile`
- Dependencies: `requirements.txt`

---

**Table of contents**
- What this app does
- Requirements
- Installation (local)
- Run with Docker
- Usage (Streamlit UI)
- Project structure
- Data & models
- Extending / retraining
- Contributing
- License

---

What this app does
-------------------
This project provides tools and an interface to:

- Explore historical forex price data and feature transforms.
- Train and evaluate forecasting models (ARIMA, gradient boosting, RNNs).
- Load pre-trained models and run live predictions from the Streamlit UI.

The Streamlit app (`app.py`) is intended for demonstration and experiment reproducibility rather than production deployment.

Requirements
------------
- Python 3.8+ recommended
- See `requirements.txt` for exact package versions. Typical core packages include `streamlit`, `pandas`, `numpy`, `scikit-learn`, `xgboost`, `tensorflow`/`torch` (depending on RNN implementation), and forecasting libraries used by the notebooks.

Installation (local)
--------------------
1. Clone the repo:

```bash
git clone <repo-url>
cd Forex-Prediction
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# or cmd
.\.venv\Scripts\activate.bat
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app locally
----------------------------
Start the app with the simple Streamlit command:

```bash
streamlit run app.py
```

If you use a virtual environment on Windows PowerShell, activate it first:

```powershell
.\.venv\Scripts\Activate.ps1
```

When Streamlit starts, open the URL shown in the terminal (usually http://localhost:8501).

Run with Docker
---------------
Build the Docker image from the repository root:

```bash
docker build -t forex-prediction .
```

Run the container, publishing the Streamlit port (8501):

```bash
docker run --rm -p 8501:8501 forex-prediction
```

Optional: mount local `models/` and `data/` directories into the container so the app can access them without rebuilding the image (Linux/macOS example):

```bash
docker run --rm -p 8501:8501 \
	-v "$(pwd)/models:/app/models" \
	-v "$(pwd)/data:/app/data" \
	forex-prediction
```

Windows PowerShell users can mount with:

```powershell
docker run --rm -p 8501:8501 \
	-v ${PWD}\\models:/app/models \
	-v ${PWD}\\data:/app/data \
	forex-prediction
```

Then open http://localhost:8501 in your browser.

Usage (Streamlit UI)
--------------------
The app UI provides controls to:

- Select a model (ARIMA, XGBoost, RNN) if available.
- Choose input currency/time window and preprocessing options.
- Run a forecast and view plots (time series, prediction vs actual, error metrics).
- Download generated forecasts and predictions where enabled.

Project structure
-----------------
Top-level files and folders:

- `app.py`: Streamlit application and UI.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Docker image for the app.
- `arima_functions.py`, `xgboost_functions.py`, `rnn_functions.py`: Model implementations and helpers.
- `data_functions.py`: Data loading and preprocessing utilities.
- `models/`: Folder where model artifacts (pickles, weights) can be stored or loaded from.
- `rnn_scalers/`: Scalers used for RNN model preprocessing.
- `data/`: example or raw datasets used by notebooks or the app.
- `1_analysis_arima.ipynb`, `2_analysis_gradient.ipynb`, `3_analysis_rnn.ipynb`: Analysis and experiment notebooks.

Data & models
-------------
- The data is found in 'data/'. Feel free to update the data with more recent observations. Don't modify the formatting.
- Trained model artifacts can live in `models/`; the Streamlit app will try to load models from here when available.

Extending / retraining
----------------------
To retrain or add models:

1. Prepare your data using `data_functions.py` or the notebooks.
2. Implement training/experiment code in the relevant module (`arima_functions.py`, `xgboost_functions.py`, or `rnn_functions.py`) or in a new script.
3. Save trained artifacts into `models/` and scalers (for RNN) into `rnn_scalers/`.
4. Update `app.py` to include options/load paths for the new model if required.

Notebooks
---------
The included notebooks demonstrate sample analysis pipelines and experiments:

- `1_analysis_arima.ipynb` — time-series decomposition and ARIMA modeling.
- `2_analysis_gradient.ipynb` — feature engineering and gradient boosting experiments (XGBoost).
- `3_analysis_rnn.ipynb` — RNN model experiments and sequence prep.

Notes & tips
------------
- The app is experimental and intended for research and demo purposes.
- Check `requirements.txt` and the notebooks for library versions used during development.
- Large model weights and data are not included; use your own datasets or export models into `models/` before using the app.

License
-------
This repository includes a `LICENSE` file — please review it for usage and redistribution terms.

Contact / Authors
-----------------
See the repository root for author details and contact information.

---