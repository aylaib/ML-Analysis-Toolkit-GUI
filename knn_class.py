import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k
        
    def fit(self, X_train, y_train):
        """Entraîne le modèle KNN avec les données d'apprentissage."""
        self.x_train = X_train.to_numpy()
        self.y_train = y_train.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x).to_numpy()
        
    def calculate_euclidean(self, sample1, sample2):
        """Calcule la distance euclidienne entre deux échantillons."""
        return np.sqrt(np.sum((sample1 - sample2) ** 2))
    
    def nearest_neighbors(self, test_sample):
        """Trouve les k plus proches voisins d'un échantillon de test."""
        distances = []
        for i in range(len(self.x_train)):
            distance = self.calculate_euclidean(self.x_train[i], test_sample)
            distances.append((self.y_train[i], distance))
        distances.sort(key=lambda x: x[1])
        return [x[0] for x in distances[:self.k]]
    
    def predict(self, test_set):
        """Prédit les classes pour un ensemble de test."""
        predictions = []
        for test_sample in test_set.to_numpy():
            neighbors = self.nearest_neighbors(test_sample)
            prediction = max(neighbors, key=neighbors.count) if neighbors else None
            predictions.append(prediction)
        return predictions
