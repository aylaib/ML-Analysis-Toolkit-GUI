import numpy as np
from naivebayes import NaiveBayes

# 1. Création des données d'exemple
# Format: [poids(g), diamètre(cm)]
X_train = np.array([
    [150, 7],  # Pomme
    [170, 7.5],  # Pomme
    [140, 6.8],  # Pomme
    [130, 6.9],  # Pomme
    [300, 12],   # Orange
    [280, 11.5], # Orange
    [320, 12.3], # Orange
    [290, 11.8], # Orange
    [50, 4],     # Prune
    [45, 3.8],   # Prune
    [55, 4.2],   # Prune
    [48, 3.9]    # Prune
])

# Labels correspondants
y_train = np.array(['pomme', 'pomme', 'pomme', 'pomme',
                   'orange', 'orange', 'orange', 'orange',
                   'prune', 'prune', 'prune', 'prune'])

# 2. Création et entraînement du modèle
classifier = NaiveBayes()
classifier.fit(X_train, y_train)

# 3. Affichage des statistiques apprises
model_details = classifier.get_model_details()
print("\n=== Statistiques du modèle ===")
print("\nProbabilités a priori (P(classe)):")
for classe, prob in model_details['priors'].items():
    print(f"{classe}: {prob:.3f}")

print("\nStatistiques par classe:")
for classe, stats in model_details['feature_stats'].items():
    print(f"\n{classe}:")
    print(f"Moyennes [poids, diamètre]: {[f'{x:.1f}' for x in stats['mean']]}")
    print(f"Écarts-types [poids, diamètre]: {[f'{x:.1f}' for x in stats['std']]}")

# 4. Test de prédiction
X_test = np.array([
    [160, 7.2],  # Devrait être une pomme
    [310, 12.1], # Devrait être une orange
    [52, 4.1]    # Devrait être une prune
])

predictions = classifier.predict(X_test)

print("\n=== Prédictions ===")
for i, (features, pred) in enumerate(zip(X_test, predictions)):
    print(f"\nÉchantillon {i+1}:")
    print(f"Caractéristiques [poids, diamètre]: {features}")
    print(f"Prédiction: {pred}")

# 5. Démonstration détaillée pour un seul fruit
test_fruit = np.array([[160, 7.2]])  # Un seul fruit à classifier
print("\n=== Démonstration détaillée pour un fruit ===")
print("Fruit test [poids, diamètre]:", test_fruit[0])

# Calcul manuel des vraisemblances pour ce fruit
for classe in ['pomme', 'orange', 'prune']:
    stats = model_details['feature_stats'][classe]
    prior = model_details['priors'][classe]
    
    print(f"\nPour la classe {classe}:")
    print(f"P({classe}) = {prior:.3f}")
    
    # Pour chaque feature (poids et diamètre)
    for i, (mean, std) in enumerate(zip(stats['mean'], stats['std'])):
        feature_name = "poids" if i == 0 else "diamètre"
        x = test_fruit[0][i]
        likelihood = classifier._calculate_likelihood(x, mean, std)
        print(f"P({feature_name}|{classe}) = {likelihood:.6f}")
