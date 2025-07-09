from math import sqrt
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_confusion_matrix(cm, classes, k):
    """Affiche la matrice de confusion sous forme de heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Matrice de confusion pour k={k}')
    plt.ylabel('Vraie classe')
    plt.xlabel('Classe prédite')
    plt.show()

def plot_metrics(k_values, metrics_dict):
    """Affiche les courbes des différentes métriques."""
    plt.figure(figsize=(12, 8))
    for metric_name, values in metrics_dict.items():
        plt.plot(k_values, values, marker='o', label=metric_name)
    
    plt.xlabel('Valeur de K')
    plt.ylabel('Score')
    plt.title('Évolution des métriques en fonction de K')
    plt.grid(True)
    plt.legend()
    plt.show()

# Chargement et préparation des données
print("1. Chargement des données...")
file_path = '/home/ayoublb/Documents/MASTER 2/FD/TP/M2/diabetes.arff'
data, meta = arff.loadarff(file_path)
df = pd.DataFrame(data)

# Division des données
print("2. Division des données...")
X = df.drop(columns='class')
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

print(f"Taille de l'ensemble d'apprentissage : {X_train.shape[0]}")
print(f"Taille de l'ensemble de test : {X_test.shape[0]}")

# Initialisation des dictionnaires pour stocker les métriques
metrics = {
    'Précision': [],
    'Rappel': [],
    'F1-Score': [],
    'Accuracy': []
}

# Test pour différentes valeurs de K
print("\n3. Évaluation du modèle pour différentes valeurs de K...")
k_values = range(1, 11)

for k in k_values:
    print(f"\nAnalyse pour K = {k}")
    model = KNN(k)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Conversion des prédictions et des vraies valeurs
    y_test_str = y_test.apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    predictions_str = [p.decode('utf-8') if isinstance(p, bytes) else p for p in predictions]
    
    # Calcul des métriques
    cm = confusion_matrix(y_test_str, predictions_str)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)
    
    # Calcul et stockage des métriques
    precision = precision_score(y_test_str, predictions_str, average='weighted', zero_division=1)
    recall = recall_score(y_test_str, predictions_str, average='weighted', zero_division=1)
    f1 = f1_score(y_test_str, predictions_str, average='weighted', zero_division=1)
    accuracy = accuracy_score(y_test_str, predictions_str)
    
    metrics['Précision'].append(precision)
    metrics['Rappel'].append(recall)
    metrics['F1-Score'].append(f1)
    metrics['Accuracy'].append(accuracy)
    
    # Affichage des résultats
    print(f"TP: {TP.sum()}, TN: {TN.sum()}, FP: {FP.sum()}, FN: {FN.sum()}")
    print(f"Précision: {precision:.3f}")
    print(f"Rappel: {recall:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Affichage de la matrice de confusion
    if k in [1, 5, 10]:  # Afficher seulement pour certaines valeurs de k
        plot_confusion_matrix(cm, np.unique(y_test_str), k)

# Affichage des courbes de métriques
plot_metrics(k_values, metrics)

# Détermination de la meilleure valeur de K
best_k_accuracy = k_values[np.argmax(metrics['Accuracy'])]
print(f"\nMeilleure valeur de K (basée sur l'accuracy) : {best_k_accuracy}")
print(f"Accuracy maximale : {max(metrics['Accuracy']):.3f}")

