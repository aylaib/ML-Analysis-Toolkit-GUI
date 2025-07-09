# Import des bibliothèques nécessaires
import numpy as np  # Pour les opérations mathématiques et matricielles
from collections import defaultdict  # Pour créer des dictionnaires avec valeurs par défaut
from sklearn.preprocessing import LabelEncoder  # Pour encoder les labels catégoriels en numériques
from typing import Tuple, Dict, Any  # Pour le typage statique des fonctions

class NaiveBayes:
    """
    Implémentation du classifieur Bayésien Naïf avec distribution gaussienne.
    
    Cette classe implémente l'algorithme de classification Naive Bayes en supposant
    que les features suivent une distribution normale (gaussienne) pour chaque classe.
    Idéal pour les données continues.
    """
    
    def __init__(self):
        """
        Initialise les attributs du classifieur.
        - class_priors: stocke les probabilités a priori P(y) pour chaque classe
        - feature_stats: stocke les statistiques (moyenne, écart-type) des features par classe
        - classes: liste des classes uniques
        - label_encoder: pour convertir les labels en format numérique
        - n_features: nombre de features
        - fitted: indique si le modèle a été entraîné
        """
        self.class_priors = {}  # Stockage des probabilités a priori P(y)
        self.feature_stats = {}  # Stockage des stats (moyenne, écart-type) par feature et classe
        self.classes = None  # Liste des classes uniques
        self.label_encoder = LabelEncoder()  # Pour encoder les labels en numérique
        self.n_features = None  # Nombre de features
        self.fitted = False  # Flag indiquant si le modèle est entraîné
    
    def _validate_input(self, X: np.ndarray, y: np.ndarray = None) -> None:
        """
        Valide le format et la cohérence des données d'entrée.
        
        Args:
            X: Matrice des features (n_échantillons × n_features)
            y: Vecteur des labels (optionnel)
            
        Raises:
            ValueError: Si les données ne respectent pas le format attendu
        """
        if X.ndim != 2:
            raise ValueError("X doit être un tableau 2D de shape (n_samples, n_features)")
        
        if y is not None:
            if len(X) != len(y):
                raise ValueError("X et y doivent avoir le même nombre d'échantillons")
                
        if self.fitted and X.shape[1] != self.n_features:
            raise ValueError(f"Nombre de features incorrect. Attendu: {self.n_features}, Reçu: {X.shape[1]}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NaiveBayes':
        """
        Entraîne le modèle sur les données fournies.
        
        Cette méthode:
        1. Valide les données d'entrée
        2. Encode les labels en format numérique
        3. Calcule les probabilités a priori des classes
        4. Calcule les statistiques (moyenne, écart-type) pour chaque feature par classe
        
        Args:
            X: Matrice des features d'entraînement
            y: Vecteur des labels d'entraînement
            
        Returns:
            self: Le classifieur entraîné (pour le chaînage des méthodes)
        """
        # Conversion en array numpy et validation
        X = np.asarray(X, dtype=np.float64)
        self._validate_input(X, y)
        
        # Encodage des labels en format numérique
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Récupération des dimensions et classes uniques
        n_samples, n_features = X.shape
        self.classes = np.unique(y_encoded)
        self.n_features = n_features
        
        # Calcul des probabilités a priori P(y) pour chaque classe
        for c in self.classes:
            self.class_priors[c] = np.sum(y_encoded == c) / n_samples
            
        # Calcul des statistiques par classe et feature
        for c in self.classes:
            class_samples = X[y_encoded == c]  # Sélection des échantillons de la classe
            self.feature_stats[c] = {
                'mean': np.mean(class_samples, axis=0),  # Moyenne par feature
                'std': np.std(class_samples, axis=0) + 1e-6  # Écart-type + epsilon pour éviter div/0
            }
        
        self.fitted = True  # Marque le modèle comme entraîné
        return self
    
    def _calculate_likelihood(self, x: float, mean: float, std: float) -> float:
        """
        Calcule la vraisemblance P(x|y) selon la loi normale.
        
        Implémente la formule de la densité de probabilité gaussienne:
        P(x|y) = 1/(sqrt(2π)σ) * exp(-(x-μ)²/(2σ²))
        
        Args:
            x: Valeur de la feature
            mean: Moyenne (μ) pour cette feature dans cette classe
            std: Écart-type (σ) pour cette feature dans cette classe
            
        Returns:
            float: La vraisemblance calculée
        """
        exponent = -((x - mean) ** 2) / (2 * (std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(exponent)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Prédit les classes pour de nouvelles données.
        
        Pour chaque échantillon:
        1. Calcule le score de chaque classe (log(P(y)) + Σ log(P(x|y)))
        2. Sélectionne la classe avec le score maximum
        
        Args:
            X: Matrice des features de test
            
        Returns:
            np.ndarray: Vecteur des prédictions de classes
            
        Raises:
            ValueError: Si le modèle n'est pas entraîné
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant de faire des prédictions")
            
        X = np.asarray(X, dtype=np.float64)
        self._validate_input(X)
        
        # Gestion du cas où X est un vecteur unique
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        predictions_encoded = []
        
        # Pour chaque échantillon
        for x in X:
            class_scores = {}  # Stocke les scores par classe
            
            # Pour chaque classe
            for c in self.classes:
                # Initialisation avec le log de la probabilité a priori
                score = np.log(self.class_priors[c])
                
                # Récupération des stats de la classe
                means = self.feature_stats[c]['mean']
                stds = self.feature_stats[c]['std']
                
                # Calcul des vraisemblances pour chaque feature
                for feature_idx in range(self.n_features):
                    likelihood = self._calculate_likelihood(
                        float(x[feature_idx]), 
                        float(means[feature_idx]), 
                        float(stds[feature_idx])
                    )
                    score += np.log(likelihood + 1e-10)  # Ajout du log de la vraisemblance
                
                class_scores[c] = score
            
            # Sélection de la classe avec le score maximum
            predictions_encoded.append(max(class_scores.items(), key=lambda x: x[1])[0])
        
        # Conversion des prédictions en labels originaux
        return self.label_encoder.inverse_transform(predictions_encoded)
    
    def get_model_details(self) -> Dict[str, Any]:
        """
        Retourne les paramètres appris par le modèle.
        
        Returns:
            Dict contenant:
            - priors: Les probabilités a priori par classe
            - feature_stats: Les statistiques (moyenne, écart-type) par feature et classe
            
        Raises:
            ValueError: Si le modèle n'est pas entraîné
        """
        if not self.fitted:
            raise ValueError("Le modèle doit être entraîné avant d'accéder aux détails")
            
        # Conversion des index numériques en labels originaux
        return {
            'priors': {
                self.label_encoder.inverse_transform([c])[0]: float(prior)
                for c, prior in self.class_priors.items()
            },
            'feature_stats': {
                self.label_encoder.inverse_transform([c])[0]: {
                    'mean': self.feature_stats[c]['mean'].tolist(),
                    'std': self.feature_stats[c]['std'].tolist()
                }
                for c in self.classes
            }
        }