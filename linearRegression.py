import numpy as np
from sklearn.preprocessing import StandardScaler
from metrics_calculator import calculate_metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000, threshold=0.5):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.threshold = threshold
        self.weights = None
        self.bias = None
        self.scaler = StandardScaler()
        self.training_loss_history = []
        self.pca = PCA(n_components=2)

    def fit(self, X, y):
        # Normalisation des données
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialisation des paramètres
        n_samples, n_features = X_scaled.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Conversion des labels en numérique
        y_numeric = np.array([1 if label == b'tested_positive' else 0 for label in y])
        
        # Entraînement avec descente de gradient
        for epoch in range(self.epochs):
            # Calcul des prédictions
            linear_pred = np.dot(X_scaled, self.weights) + self.bias
            predictions = self._sigmoid(linear_pred)
            
            # Calcul des gradients
            dw = (1/n_samples) * np.dot(X_scaled.T, (predictions - y_numeric))
            db = (1/n_samples) * np.sum(predictions - y_numeric)
            
            # Mise à jour des paramètres
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Calcul et stockage de la perte
            loss = self._binary_cross_entropy(y_numeric, predictions)
            self.training_loss_history.append(loss)

        # Appliquer PCA pour la visualisation
        self.X_pca = self.pca.fit_transform(X_scaled)
        self.y = y_numeric

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        linear_pred = np.dot(X_scaled, self.weights) + self.bias
        predictions = self._sigmoid(linear_pred)
        return np.array([b'tested_positive' if p > self.threshold else b'tested_negative' for p in predictions])

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _binary_cross_entropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def get_model_details(self):
        return {
            'learning_rate': self.learning_rate,
            'epochs': self.epochs,
            'threshold': self.threshold,
            'feature_weights': list(zip(range(len(self.weights)), self.weights)),
            'bias': self.bias,
            'final_loss': self.training_loss_history[-1] if self.training_loss_history else None
        }

    def visualize_decision_boundary(self):
        """Visualise la frontière de décision et les données"""
        plt.figure(figsize=(15, 5))

        # 1. Graphique de la perte pendant l'entraînement
        plt.subplot(131)
        plt.plot(self.training_loss_history)
        plt.title('Évolution de la perte')
        plt.xlabel('Époque')
        plt.ylabel('Perte')

        # 2. Visualisation des données et de la frontière de décision
        plt.subplot(132)
        
        # Créer une grille pour tracer la frontière de décision
        x_min, x_max = self.X_pca[:, 0].min() - 1, self.X_pca[:, 0].max() + 1
        y_min, y_max = self.X_pca[:, 1].min() - 1, self.X_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                            np.linspace(y_min, y_max, 100))
        
        # Transformer les points de la grille dans l'espace original
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_points_original = self.pca.inverse_transform(grid_points)
        
        # Prédire les classes pour chaque point de la grille
        Z = self.predict(self.scaler.inverse_transform(grid_points_original))
        Z = np.array([1 if pred == b'tested_positive' else 0 for pred in Z]).reshape(xx.shape)
        
        # Tracer la frontière de décision
        plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
        
        # Tracer les points de données
        scatter = plt.scatter(self.X_pca[:, 0], self.X_pca[:, 1], 
                            c=self.y, cmap='RdYlBu', 
                            edgecolor='black', linewidth=1, alpha=0.7)
        plt.colorbar(scatter)
        plt.title('Frontière de décision')
        plt.xlabel('Première composante principale')
        plt.ylabel('Deuxième composante principale')

        # 3. Distribution des probabilités prédites
        plt.subplot(133)
        X_scaled = self.scaler.transform(self.pca.inverse_transform(self.X_pca))
        probs = self._sigmoid(np.dot(X_scaled, self.weights) + self.bias)
        plt.hist(probs[self.y == 0], bins=20, alpha=0.5, label='Négatif', color='blue')
        plt.hist(probs[self.y == 1], bins=20, alpha=0.5, label='Positif', color='red')
        plt.axvline(x=self.threshold, color='black', linestyle='--', label='Seuil')
        plt.title('Distribution des probabilités')
        plt.xlabel('Probabilité prédite')
        plt.ylabel('Nombre d échantillons')
        plt.legend()

        plt.tight_layout()
        plt.show()