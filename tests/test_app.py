"""Smoke-test the Streamlit app end to end with Streamlit's AppTest."""
import json
import os
import shutil

import pytest

ROOT = os.path.join(os.path.dirname(__file__), '..')
APP = os.path.join(ROOT, 'bed_management_app.py')
DB = os.path.join(ROOT, 'patients_db.json')

st_testing = pytest.importorskip('streamlit.testing.v1')


@pytest.fixture
def clean_db():
    backup = DB + '.bak'
    had_db = os.path.exists(DB)
    if had_db:
        shutil.move(DB, backup)
    yield
    if os.path.exists(DB):
        os.remove(DB)
    if had_db:
        shutil.move(backup, DB)


def test_app_renders_and_admits_patient(clean_db):
    at = st_testing.AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception

    at.sidebar.button[0].click().run()   # "PREDICT LOS & ADMIT"
    assert not at.exception

    with open(DB) as f:
        saved = json.load(f)
    assert len(saved) == 1
    assert saved[0]['patient_id'] == 'NEW-001'
    assert 1 <= saved[0]['length_of_stay'] <= 30
