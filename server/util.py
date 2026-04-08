import json
import pickle
import numpy as np

__locations = None
__data_columns = None
__model = None


def get_estimated_price(location, sqft, bhk, bath):
    try:
        loc_index = __data_columns.index(location.lower())
    except:
        loc_index = -1
    print('running')
    x = np.zeros(len(__data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    # print(f"Input vector (length {len(x)}): {x}")  # Check length and content of x
    # print(f"Expected feature length: {__model.n_features_in_}")  # Model's expected input size

    return round(__model.predict([x])[0], 2)


def get_location_names():
    global __locations
    return __locations


def load_saved_artifacts():
    print('loading saved artifacts...start')
    global __data_columns
    global __locations
    global __model
    with open('artifacts/5columns_ma.json', 'r') as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]

    with open('artifacts/5abd_sana_model.pickle', 'rb') as f:
        __model = pickle.load(f)
    print('loading saved artifacts...done')


if __name__ == "__main__":
    load_saved_artifacts()
    print(get_location_names())
    print(get_estimated_price('Casablanca', 1000, 3, 3))
    print(get_estimated_price('Agadir', 1000, 2, 2))
    print(get_estimated_price('Marrakech', 1000, 2, 2))
    print(get_estimated_price('Tanger', 1000, 2, 2))
