import json
import pickle
import numpy as np
import os

__locations = None
__data_columns = None
__model = None

def get_estimated_price(location, sqft, bhk, bath):
    """
    Prend en entrée les caractéristiques d'une maison et retourne le prix estimé.
    """
    try:
        loc_index = __data_columns.index(location.lower())
    except ValueError:
        loc_index = -1

    # Créer un vecteur de features de la même taille que les colonnes
    x = np.zeros(len(__data_columns))

    # ⚠️ Ton fichier JSON commence par ["bhk", "total_sqft", "bath", "agadir", ...]
    # Donc l'ordre est : bhk -> index 0, sqft -> index 1, bath -> index 2
    x[0] = bhk
    x[1] = sqft
    x[2] = bath

    # Encoder la localisation en one-hot
    if loc_index >= 0:
        x[loc_index] = 1

    return round(__model.predict([x])[0], 2)


def get_location_names():
    """
    Retourne la liste des localisations disponibles.
    """
    global __locations
    return __locations


def load_saved_artifacts():
    """
    Charge les colonnes et le modèle sauvegardés dans artifacts/.
    """
    print('loading saved artifacts...start')
    global __data_columns
    global __locations
    global __model

    # Chemin absolu vers le dossier artifacts (dans le même dossier que util.py)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.join(base_dir, "artifacts")

    # Charger les colonnes
    with open(os.path.join(artifacts_dir, '5columns_ma.json'), 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]  # les colonnes après bhk, sqft, bath

    # Charger le modèle
    with open(os.path.join(artifacts_dir, '5abd_sana_model.pickle'), 'rb') as f:
        __model = pickle.load(f)

    print('loading saved artifacts...done')


if __name__ == "__main__":
    load_saved_artifacts()
    print("Locations disponibles :", get_location_names())
    print("Casablanca 1000 sqft, 3 BHK, 3 bath ->", get_estimated_price('Casablanca', 1000, 3, 3))
    print("Agadir 1000 sqft, 2 BHK, 2 bath ->", get_estimated_price('Agadir', 1000, 2, 2))
    print("Marrakech 1000 sqft, 2 BHK, 2 bath ->", get_estimated_price('Marrakech', 1000, 2, 2))
    print("Tanger 1000 sqft, 2 BHK, 2 bath ->", get_estimated_price('Tanger', 1000, 2, 2))
