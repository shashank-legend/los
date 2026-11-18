# MediCore BMS — Hospital Bed Management with LOS Prediction

A Streamlit dashboard that predicts a new patient's **Length of Stay (LOS)** with
machine-learning models and uses it to manage hospital beds: live bed map per
department, discharge forecast, and occupancy analytics.

## Features

- **LOS prediction** from 27 admission-time inputs (demographics, clinical
  status, vitals, surgery, history) with 5 models: Linear/Ridge Regression,
  Decision Tree, Random Forest, Gradient Boosting.
- **Live bed map** for 20 departments / 200 beds, colour-coded by status
  (critical, infected, post-surgery, discharging, ML-admitted).
- **Capacity check**: admission is blocked when the chosen department is full,
  and the free-bed count is shown while filling the form.
- **Discharge forecast** for the next 7 days, 14-day occupancy trend,
  department occupancy chart.
- **Patient management**: ML-admitted patients persist in `patients_db.json`;
  discharge individual patients or export the list to CSV.
- **Model metrics** (MAE, RMSE, R², MAPE) computed on the held-out test set.

## Model performance (test set, 1,000 patients)

| Model | MAE (days) | RMSE | R² | MAPE |
|---|---|---|---|---|
| Linear Regression | 1.518 | 1.905 | 0.460 | 27.4% |
| Ridge Regression | 1.518 | 1.905 | 0.460 | 27.4% |
| Gradient Boosting | 1.560 | 1.958 | 0.430 | 28.2% |
| Random Forest | 1.599 | 1.992 | 0.410 | 29.3% |
| Decision Tree | 1.729 | 2.136 | 0.321 | 30.6% |

## Quick start

```bash
pip install -r requirements.txt
streamlit run bed_management_app.py
```

Opens at http://localhost:8501. The app can be started from any directory.

## Retraining

```bash
python preprocessing.py   # encode, scale, select features, split → X_/y_*.csv, scaler.pkl
python save_models.py     # train 5 models, evaluate, write model_bundle.pkl
```

`model_bundle.pkl` is tied to the pinned scikit-learn version in
`requirements.txt`; retrain if you upgrade scikit-learn.

## Tests

```bash
python -m pytest -v
```

Covers feature construction (app inputs must match the training features
exactly), prediction accuracy on real rows, every model, LOS categories, and an
end-to-end Streamlit run that admits a patient.

## Project structure

| File | Purpose |
|---|---|
| `bed_management_app.py` | Streamlit dashboard |
| `los_model.py` | Load model bundle, build features, predict LOS |
| `preprocessing.py` | Data cleaning, encoding, scaling, feature selection, split |
| `save_models.py` | Train, evaluate and save models |
| `_loss.py` | scikit-learn unpickling compatibility shim |
| `synthetic_hospital_data.csv` | 5,000 synthetic patient records |
| `hospital_occupancy_timeseries.csv` | Daily occupancy history |
| `tests/` | pytest suite |

## Changelog

**v2**
- Fixed LOS predictions in the app. Training data was scaled in
  `preprocessing.py`, but the app re-scaled raw form inputs with a scaler fitted
  on that already-scaled data, so models received unscaled values. With Linear
  Regression the app's average error on real patients was ~14 days (many
  predictions stuck at the 30-day cap); it is now ~1.5 days. The raw-data scaler
  is now saved and reused.
- Metrics are computed on the test set at training time instead of hard-coded.
- Department capacity check, free-bed indicator, discharge and CSV export.
- Admission date defaults to the dashboard date; new patient IDs never collide.
- Data files resolve relative to the script, so the app runs from any folder.
- Added `los_model.py`, pytest suite and this README.
