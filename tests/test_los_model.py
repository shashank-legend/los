"""
Tests for the LOS prediction pipeline.

Run with:  python -m pytest -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from los_model import (load_bundle, build_feature_row, predict_los,  # noqa: E402
                       los_category, MIN_LOS, MAX_LOS)

HOUR_MAP = {'09:00-12:00': 0, '12:00-15:00': 1, '15:00-18:00': 2, '18:00-21:00': 3}


@pytest.fixture(scope='module')
def bundle():
    return load_bundle()


@pytest.fixture(scope='module')
def raw_df():
    return pd.read_csv(os.path.join(ROOT, 'synthetic_hospital_data.csv'))


@pytest.fixture(scope='module')
def preprocessed():
    return pd.read_csv(os.path.join(ROOT, 'preprocessed_full.csv'))


def inp_from_raw(r):
    """Convert a raw dataset row into the dict the app's form produces."""
    return {
        'age': r.age, 'gender': r.gender, 'marital': r.marital_status,
        'ses': r.socioeconomic_status, 'insurance': r.insurance_type,
        'admission_type': r.admission_type,
        'adm_date': pd.Timestamp(r.admission_date),
        'hour': HOUR_MAP[r.hour_of_admission],
        'severity': r.severity_of_illness, 'comorbidity': r.has_comorbidity,
        'charlson': r.charlson_comorbidity_index, 'infection': r.has_infection,
        'surgery': r.surgery_performed, 'surg_dur': r.surgery_duration_mins,
        'n_proc': r.num_procedures, 'n_meds': r.num_medications,
        'wt_loss': r.weight_loss, 'diet': r.dietary_pattern_change,
        'func': r.functional_capacity_change, 'malnut': r.malnutrition,
        'bmi': r.bmi, 'sbp': r.systolic_bp, 'dbp': r.diastolic_bp,
        'temp': r.body_temperature, 'glucose': r.blood_glucose,
        'prev_adm': r.previous_admissions, 'op_visits': r.outpatient_visits_30d,
    }


def default_inp():
    return {
        'age': 50, 'gender': 'Male', 'marital': 'Married', 'ses': 'Middle',
        'insurance': 'Government', 'admission_type': 'Elective',
        'adm_date': pd.Timestamp('2026-03-10'), 'hour': 0, 'severity': 2,
        'comorbidity': 0, 'charlson': 0, 'infection': 0, 'surgery': 0,
        'surg_dur': 0, 'n_proc': 1, 'n_meds': 3, 'wt_loss': 0, 'diet': 0,
        'func': 0, 'malnut': 0, 'bmi': 25.0, 'sbp': 120, 'dbp': 80,
        'temp': 37.0, 'glucose': 100, 'prev_adm': 0, 'op_visits': 2,
    }


def test_bundle_has_expected_keys(bundle):
    for k in ('models', 'feature_cols', 'scaler', 'scale_cols', 'metrics'):
        assert k in bundle
    assert set(bundle['metrics']) == set(bundle['models'])


def test_app_features_match_training_features(bundle, raw_df, preprocessed):
    """
    The app must turn raw inputs into exactly the scaled values the model was
    trained on. This is the regression test for the double-scaling bug.
    """
    feats = [c for c in bundle['feature_cols'] if c != 'nosocomial_infection']
    for i in range(25):
        r = raw_df.iloc[i]
        X = build_feature_row(inp_from_raw(r), bundle)
        expected = preprocessed.loc[i, feats].astype(float).values
        np.testing.assert_allclose(X[feats].values[0], expected, atol=1e-6,
                                   err_msg=f"row {i} ({r.patient_id})")


def test_predictions_track_actual_los(bundle, raw_df):
    """Predicted LOS on real rows should be close to the recorded LOS."""
    sample = raw_df.sample(200, random_state=0)
    preds = [predict_los(inp_from_raw(r), 'Linear Regression', bundle)
             for r in sample.itertuples()]
    mae = np.mean(np.abs(np.array(preds) - sample.length_of_stay.values))
    assert mae < 2.5, f"MAE {mae:.2f} — features are probably mis-scaled"


@pytest.mark.parametrize('model', ['Linear Regression', 'Ridge Regression',
                                   'Decision Tree', 'Random Forest',
                                   'Gradient Boosting'])
def test_every_model_predicts_in_range(bundle, model):
    los = predict_los(default_inp(), model, bundle)
    assert MIN_LOS <= los <= MAX_LOS


def test_sicker_patient_stays_longer(bundle):
    mild = default_inp()
    severe = dict(mild, severity=3, comorbidity=1, charlson=6, infection=1,
                  surgery=1, surg_dur=240, n_proc=5, n_meds=12, malnut=1,
                  admission_type='Emergency', age=80)
    assert (predict_los(severe, 'Linear Regression', bundle)
            > predict_los(mild, 'Linear Regression', bundle))


@pytest.mark.parametrize('days,cat', [(1, 'Short (1-3d)'), (3, 'Short (1-3d)'),
                                      (4, 'Medium (4-7d)'), (8, 'Long (8-14d)'),
                                      (15, 'Very Long (>14d)')])
def test_los_category(days, cat):
    assert los_category(days) == cat
