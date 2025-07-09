import pandas as pd
from scipy.io import arff
from sklearn.model_selection import train_test_split
import os

def validate_file_path(file_path):
    """Validates and processes the file path input."""
    if not isinstance(file_path, str):
        raise ValueError(f"Le chemin du fichier doit être une chaîne de caractères, pas {type(file_path)}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier '{file_path}' n'existe pas. Veuillez vérifier le chemin.")
    
    if not file_path.lower().endswith('.arff'):
        raise ValueError("Le fichier doit être au format ARFF")

def load_data_safe(file_path):
    """Version sécurisée du chargeur de données qui inclut la validation."""
    try:
        validate_file_path(file_path)
        print("Chargement des données...")
        data, meta = arff.loadarff(file_path)
        return pd.DataFrame(data), meta
    except Exception as e:
        print(f"Erreur lors du chargement des données: {str(e)}")
        print("Assurez-vous que:")
        print("1. Le chemin du fichier est correct")
        print("2. Le fichier existe")
        print("3. Le fichier est au format ARFF valide")
        raise

def prepare_data(df, test_size=0.2, random_state=42):
    """Prépare les données pour l'apprentissage."""
    print("Préparation des données...")
    
    if df is None or df.empty:
        raise ValueError("Le DataFrame est vide ou None")
        
    if 'class' not in df.columns:
        raise ValueError("La colonne 'class' n'est pas présente dans les données")
    
    X = df.drop(columns='class')
    y = df['class']
    
    # Division en ensembles d'apprentissage et de test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    print(f"Taille de l'ensemble d'apprentissage : {X_train.shape[0]}")
    print(f"Taille de l'ensemble de test : {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

# Pour maintenir la compatibilité avec le code existant
def load_data(file_path):
    """Fonction wrapper pour maintenir la compatibilité avec le code existant."""
    return load_data_safe(file_path)