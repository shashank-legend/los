"""
los_model.py
------------
Loading the trained model bundle and turning a new patient's raw
admission details into a Length-of-Stay (LOS) prediction.

Kept separate from the Streamlit app so it can be unit-tested and
reused (e.g. from a notebook or an API).
"""
import os
import pickle
from datetime import date

import numpy as np
import pandas as pd

import _loss  # noqa: F401  (sklearn unpickling compatibility shim)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUNDLE_PATH = os.path.join(SCRIPT_DIR, 'model_bundle.pkl')

INSURANCE_MAP = {'Uninsured': 0, 'Government': 1, 'Private': 2}
SES_MAP = {'Low': 0, 'Middle': 1, 'High': 2}
MARITAL_MAP = {'Single': 0, 'Married': 1, 'Divorced': 2, 'Widowed': 3}

MIN_LOS, MAX_LOS = 1, 30


def load_bundle(path=BUNDLE_PATH):
    with open(path, 'rb') as f:
        bundle = pickle.load(f)
    missing = {'models', 'feature_cols', 'scaler', 'scale_cols'} - set(bundle)
    if missing:
        raise ValueError(
            f"model_bundle.pkl is missing {sorted(missing)}. "
            "Re-run `python preprocessing.py && python save_models.py`.")
    return bundle


def raw_features(inp):
    """Map the form inputs to the un-scaled feature names used in training."""
    adm = inp['adm_date']
    if not isinstance(adm, (date, pd.Timestamp)):
        adm = pd.Timestamp(adm)
    return {
        'age':                        inp['age'],
        'gender':                     1 if inp['gender'] == 'Female' else 0,
        'admission_type':             1 if inp['admission_type'] == 'Emergency' else 0,
        'severity_of_illness':        inp['severity'],
        'has_comorbidity':            inp['comorbidity'],
        'charlson_comorbidity_index': inp['charlson'],
        'has_infection':              inp['infection'],
        'nosocomial_infection':       0,  # unknown at admission
        'surgery_performed':          inp['surgery'],
        'surgery_duration_mins':      inp['surg_dur'],
        'num_procedures':             inp['n_proc'],
        'num_medications':            inp['n_meds'],
        'weight_loss':                inp['wt_loss'],
        'dietary_pattern_change':     inp['diet'],
        'functional_capacity_change': inp['func'],
        'malnutrition':               inp['malnut'],
        'bmi':                        inp['bmi'],
        'systolic_bp':                inp['sbp'],
        'diastolic_bp':               inp['dbp'],
        'body_temperature':           inp['temp'],
        'blood_glucose':              inp['glucose'],
        'previous_admissions':        inp['prev_adm'],
        'outpatient_visits_30d':      inp['op_visits'],
        'admission_month':            adm.month,
        'admission_year':             adm.year,
        'admission_quarter':          (adm.month - 1) // 3 + 1,
        'weekend_admission':          1 if adm.weekday() >= 5 else 0,
        'hour_of_admission':          inp['hour'],
        'insurance_type':             INSURANCE_MAP[inp['insurance']],
        'socioeconomic_status':       SES_MAP[inp['ses']],
        'marital_status':             MARITAL_MAP[inp['marital']],
    }


def build_feature_row(inp, bundle):
    """
    Build the model-ready, single-row DataFrame.

    Continuous columns are scaled with the SAME scaler that was fitted on the
    raw training data in preprocessing.py. Any scaled column we don't collect
    from the form (e.g. outpatient_visits_180d) is filled with its training
    mean, which becomes 0 after scaling.
    """
    raw = raw_features(inp)
    scaler, scale_cols = bundle['scaler'], bundle['scale_cols']
    means = dict(zip(scale_cols, scaler.mean_))

    to_scale = pd.DataFrame([{c: raw.get(c, means[c]) for c in scale_cols}])
    scaled = dict(zip(scale_cols, scaler.transform(to_scale)[0]))

    row = {}
    for c in bundle['feature_cols']:
        if c in scaled:
            row[c] = scaled[c]
        else:
            row[c] = raw.get(c, 0)
    return pd.DataFrame([row])[bundle['feature_cols']]


def predict_los(inp, model_name, bundle):
    X = build_feature_row(inp, bundle)
    pred = bundle['models'][model_name].predict(X)[0]
    return int(np.clip(round(pred), MIN_LOS, MAX_LOS))


def los_category(days):
    if days <= 3:
        return 'Short (1-3d)'
    if days <= 7:
        return 'Medium (4-7d)'
    if days <= 14:
        return 'Long (8-14d)'
    return 'Very Long (>14d)'
