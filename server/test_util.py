import os
import util
import subprocess

def setup_module(module):
    # Si les artefacts n'existent pas, lancer l'entraînement
    if not os.path.exists("artifacts/5columns_ma.json") or not os.path.exists("artifacts/5abd_sana_model.pickle"):
        subprocess.run(["python", "src/train.py"], check=True)

def test_load_artifacts():
    util.load_saved_artifacts()
    assert len(util.get_location_names()) > 0

def test_prediction():
    util.load_saved_artifacts()
    price = util.get_estimated_price("casablanca", 1000, 3, 2)
    assert isinstance(price, float)
    assert price > 0
