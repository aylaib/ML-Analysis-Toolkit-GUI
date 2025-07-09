import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

class SupportVectorMachine:
    def __init__(self, kernel='rbf', C=1.0):
        """
        Initialise le classifieur SVM
        
        Args:
            kernel (str): Type de kernel (défaut: 'rbf')
            C (float): Paramètre de régularisation (défaut: 1.0)
        """
        self.model = SVC(kernel=kernel, C=C, random_state=123)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def fit(self, X_train, y_train):
        """
        Entraîne le modèle SVM
        
        Args:
            X_train (np.array): Données d'entraînement
            y_train (np.array): Labels d'entraînement
        """
        # Convertir les labels en chaînes de caractères si ce sont des bytes
        if y_train.dtype.kind == 'S':
            y_train = y_train.astype(str)
        
        # Encoder les labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        # Normalisation des données
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Entraînement du modèle
        self.model.fit(X_train_scaled, y_train_encoded)
    
    def predict(self, X_test):
        """
        Fait des prédictions sur les données de test
        
        Args:
            X_test (np.array): Données de test
        
        Returns:
            np.array: Prédictions
        """
        # Normalisation des données de test
        X_test_scaled = self.scaler.transform(X_test)
        
        # Prédiction
        predictions_encoded = self.model.predict(X_test_scaled)
        
        # Décodage des prédictions
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def get_model_details(self):
        """
        Récupère les détails du modèle SVM
        
        Returns:
            dict: Détails du modèle
        """
        details = {
            'kernel': self.model.kernel,
            'C': self.model.C,
            'support_vectors_count': self.model.support_vectors_.shape[0],
            'classes': self.label_encoder.classes_
        }
        return details