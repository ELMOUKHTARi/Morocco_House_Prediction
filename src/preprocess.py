import pandas as pd
import numpy as np

def load_data(path):
    """
    Charger les données depuis un fichier CSV
    """
    df = pd.read_csv(path)
    return df


def convert_sqft_to_num(x):
    """
    Convertir total_sqft en nombre (gestion des cas '1000-1200')
    """
    try:
        if '-' in str(x):
            tokens = x.split('-')
            return (float(tokens[0]) + float(tokens[1])) / 2
        return float(x)
    except:
        return None


def preprocess_data(df):
    """
    Nettoyage et préparation des données
    """

    # Supprimer les valeurs nulles importantes
    df = df.dropna(subset=['location', 'total_sqft', 'bath', 'bhk', 'price'])

    # Convertir total_sqft
    df['total_sqft'] = df['total_sqft'].apply(convert_sqft_to_num)

    # Supprimer valeurs nulles après conversion
    df = df.dropna(subset=['total_sqft'])

    # Nettoyer location
    df['location'] = df['location'].apply(lambda x: x.strip().lower())

    # One Hot Encoding
    dummies = pd.get_dummies(df['location'])

    df = pd.concat([df, dummies], axis=1)

    # Supprimer colonne location
    df = df.drop('location', axis=1)

    return df


def split_data(df):
    """
    Séparer features et target
    """
    X = df.drop('price', axis=1)
    y = df['price']
    return X, y