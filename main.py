import sys
from data_processor import load_data, prepare_data
from knn_class import KNN
from visualization import plot_confusion_matrix, plot_metrics
from metrics_calculator import calculate_metrics
from naivebayes import NaiveBayes
from decision_tree import DecisionTreeClassifier
from neuralnetwork import run_neural_network_analysis  
from svm import SupportVectorMachine
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from linearRegression import LinearRegression
from apriori import Apriori

def print_menu():
    """Affiche le menu principal."""
    print("\n=== Menu Principal ===")
    print("1. Appliquer KNN")
    print("2. Appliquer Naive Bayes")
    print("3. Appliquer Arbre de Décision")
    print("4. Appliquer Réseau de Neurones")
    print("5. Appliquer SVM")
    print("6. Appliquer Régression Linéaire")
    #print("7. Appliquer Apriori")
    print("8. Quitter")
    return input("Choisissez une option (1-8): ")

def run_knn_analysis():
    """Exécute l'analyse KNN complète."""
    # Chargement des données
    file_path = input("Entrez le chemin du fichier ARFF (par défaut: 'diabetes.arff'): ") or '/home/ayoublb/Documents/MASTER 2/FD/TP/M2/diabetes.arff'
    df, meta = load_data(file_path)
    
    # Préparation des données
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Initialisation des métriques
    metrics_history = {
        'Précision': [],
        'Rappel': [],
        'F1-Score': [],
        'Accuracy': []
    }
    
    # Test pour différentes valeurs de K
    k_values = range(1, 11)
    for k in k_values:
        print(f"\nAnalyse pour K = {k}")
        
        # Création et entraînement du modèle
        model = KNN(k)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Calcul des métriques
        metrics = calculate_metrics(y_test, predictions)
        
        # Stockage des métriques
        metrics_history['Précision'].append(metrics['precision'])
        metrics_history['Rappel'].append(metrics['recall'])
        metrics_history['F1-Score'].append(metrics['f1'])
        metrics_history['Accuracy'].append(metrics['accuracy'])
        
        # Affichage des résultats
        print(f"TP: {metrics['TP']}, TN: {metrics['TN']}, FP: {metrics['FP']}, FN: {metrics['FN']}")
        print(f"Précision: {metrics['precision']:.3f}")
        print(f"Rappel: {metrics['recall']:.3f}")
        print(f"F1-Score: {metrics['f1']:.3f}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        
        # Affichage de la matrice de confusion pour certaines valeurs de k
        if k in [1, 5, 10]:
            plot_confusion_matrix(metrics['confusion_matrix'], ['tested_negative', 'tested_positive'], k)
    
    # Affichage des courbes de métriques
    plot_metrics(k_values, metrics_history)
    
    # Détermination de la meilleure valeur de K
    best_k_accuracy = k_values[max(range(len(k_values)), 
                                 key=lambda i: metrics_history['Accuracy'][i])]
    print(f"\nMeilleure valeur de K (basée sur l'accuracy) : {best_k_accuracy}")
    print(f"Accuracy maximale : {max(metrics_history['Accuracy']):.3f}")

def run_naive_bayes_analysis():
    """Exécute l'analyse avec le classifieur Bayésien Naïf."""
    # Chargement des données
    file_path = input("Entrez le chemin du fichier ARFF (par défaut: 'diabetes.arff'): ") or '/home/ayoublb/Documents/MASTER 2/FD/TP/M2/diabetes.arff'
    print("\n=== Analyse avec le Classifieur Bayésien Naïf ===")
    print("Chargement et préparation des données...")
    df, meta = load_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    print("\nEntraînement du modèle Bayésien Naïf...")
    model = NaiveBayes()
    model.fit(X_train, y_train)
    
    # Affichage des détails du modèle
    details = model.get_model_details()
    print("\nDétails du modèle :")
    print("Probabilités a priori des classes :")
    for classe, prob in details['priors'].items():
        print(f"  - Classe {classe}: {prob:.3f}")
    
    print("\nStatistiques des features par classe :")
    feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigree', 'Age']
    for classe in details['feature_stats']:
        print(f"\nClasse {classe}:")
        print("  Moyennes:")
        for i, mean in enumerate(details['feature_stats'][classe]['mean']):
            print(f"    {feature_names[i]}: {mean:.3f}")
        print("  Écarts-types:")
        for i, std in enumerate(details['feature_stats'][classe]['std']):
            print(f"    {feature_names[i]}: {std:.3f}")
    
    # Prédictions et métriques
    print("\nCalcul des prédictions et des métriques...")
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    
    # Affichage des résultats
    print("\nRésultats de la classification :")
    print(f"Matrice de confusion :")
    print(f"TP: {metrics['TP']}, TN: {metrics['TN']}")
    print(f"FP: {metrics['FP']}, FN: {metrics['FN']}")
    print(f"\nMétriques de performance :")
    print(f"Précision: {metrics['precision']:.3f}")
    print(f"Rappel: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    
    # Affichage de la matrice de confusion
    plot_confusion_matrix(metrics['confusion_matrix'], 
                        ['tested_negative', 'tested_positive'], 
                        'Naive Bayes')
    
def run_decision_tree_analysis():
    """Exécute l'analyse avec l'arbre de décision."""
    # Chargement des données
    file_path = input("Entrez le chemin du fichier ARFF (par défaut: 'diabetes.arff'): ") or '/home/ayoublb/Documents/MASTER 2/FD/TP/M2/diabetes.arff'
    print("\n=== Analyse avec l'Arbre de Décision ===")
    print("Chargement et préparation des données...")
    df, meta = load_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Choix de la métrique
    print("\nChoisir la métrique de division:")
    print("1. Ratio de gain (gain ratio)")
    print("2. Indice de Gini (gini index)")
    metric_choice = input("Votre choix (1-2): ")
    metric = "gain_ratio" if metric_choice == "1" else "gini_index"
    
    print(f"\nEntraînement de l'arbre de décision avec {metric}...")
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train, metric=metric)
    
    # Prédictions et métriques
    print("\nCalcul des prédictions et des métriques...")
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    
    # Affichage des résultats
    print("\nRésultats de la classification :")
    print(f"Matrice de confusion :")
    print(f"TP: {metrics['TP']}, TN: {metrics['TN']}")
    print(f"FP: {metrics['FP']}, FN: {metrics['FN']}")
    print(f"\nMétriques de performance :")
    print(f"Précision: {metrics['precision']:.3f}")
    print(f"Rappel: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    
    # Affichage de la matrice de confusion
    plot_confusion_matrix(metrics['confusion_matrix'], 
                        ['tested_negative', 'tested_positive'], 
                        'Decision Tree')
    
    # Export de l'arbre en PDF
    export_choice = input("\nVoulez-vous exporter l'arbre en PDF? (o/n): ")
    if export_choice.lower() == 'o':
        filename = "decision_tree_diabetes.pdf"
        model.export_tree_pdf(filename=filename)
        print(f"Arbre exporté dans {filename}")

def visualize_svm_results(X_train, y_train, model, metrics):
    """
    Créer plusieurs visualisations pour l'analyse SVM, incluant une vue 3D
    
    Args:
        X_train (np.array): Données d'entraînement
        y_train (np.array): Labels d'entraînement
        model (SupportVectorMachine): Modèle SVM entraîné
        metrics (dict): Métriques de performance
    """
    plt.figure(figsize=(20, 5))
    
    # 1. Réduction de dimensionnalité avec PCA pour visualisation 3D
    pca = PCA(n_components=3)
    X_scaled = model.scaler.fit_transform(X_train)
    X_pca = pca.fit_transform(X_scaled)
    
    # Création du subplot 3D
    ax = plt.subplot(141, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2],
                        c=[0 if y == b'tested_negative' else 1 for y in y_train],
                        cmap='viridis', alpha=0.7)
    ax.set_title('Distribution 3D des données (PCA)')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    plt.colorbar(scatter, label='Classe')
    
    # Ajout de la surface de décision 3D
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    z_min, z_max = X_pca[:, 2].min() - 1, X_pca[:, 2].max() + 1
    
    # Création d'une grille 3D plus grossière pour la performance
    xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 20),
                            np.linspace(y_min, y_max, 20),
                            np.linspace(z_min, z_max, 20))
    
    # Préparation des points de la grille
    grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
    grid_points_original = pca.inverse_transform(grid_points)
    grid_points_scaled = model.scaler.transform(grid_points_original)
    
    # Prédiction pour la surface de décision
    Z = model.model.predict(grid_points_scaled)
    Z = Z.reshape(xx.shape)
    
    # Visualisation de la surface de décision avec un niveau de transparence
    ax.contour3D(xx[:,:,10], yy[:,:,10], Z[:,:,10], 
                 levels=[0.5], alpha=0.3, cmap='viridis')
    
    # 2. Vue 2D classique (projection sur les deux premières composantes)
    plt.subplot(142)
    scatter_2d = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                           c=[0 if y == b'tested_negative' else 1 for y in y_train], 
                           cmap='viridis', alpha=0.7)
    plt.title('Projection 2D (PC1 vs PC2)')
    plt.xlabel('Première composante principale')
    plt.ylabel('Deuxième composante principale')
    
    # 3. Plan de décision 2D
    plt.subplot(143)
    h = 0.02
    xx_2d, yy_2d = np.meshgrid(np.arange(x_min, x_max, h),
                              np.arange(y_min, y_max, h))
    
    grid_2d = np.c_[xx_2d.ravel(), yy_2d.ravel(), np.zeros_like(xx_2d.ravel())]
    grid_points_original_2d = pca.inverse_transform(grid_2d)
    grid_points_scaled_2d = model.scaler.transform(grid_points_original_2d)
    
    Z_2d = model.model.predict(grid_points_scaled_2d)
    Z_2d = Z_2d.reshape(xx_2d.shape)
    
    plt.contourf(xx_2d, yy_2d, Z_2d, alpha=0.4, cmap='viridis')
    plt.scatter(X_pca[:, 0], X_pca[:, 1], 
               c=[0 if y == b'tested_negative' else 1 for y in y_train],
               cmap='viridis', alpha=0.7)
    plt.title('Plan de Décision SVM')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    
    # 4. Matrice de confusion
    plt.subplot(144)
    sns.heatmap(metrics['confusion_matrix'], 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Négatif', 'Positif'],
                yticklabels=['Négatif', 'Positif'])
    plt.title('Matrice de Confusion')
    
    plt.tight_layout()
    plt.show()


# Ajoutez cette nouvelle fonction pour l'analyse SVM
def run_svm_analysis():
    """Exécute l'analyse avec SVM."""
    # Chargement des données
    file_path = input("Entrez le chemin du fichier ARFF (par défaut: 'diabetes.arff'): ") or '/home/ayoublb/Documents/MASTER 2/FD/TP/M2/diabetes.arff'
    print("\n=== Analyse avec Support Vector Machine ===")
    print("Chargement et préparation des données...")
    df, meta = load_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Choix du kernel
    print("\nChoisir le type de kernel:")
    print("1. Radial Basis Function (RBF)")
    print("2. Linéaire")
    print("3. Polynomial")
    kernel_choice = input("Votre choix (1-3): ")
    
    kernels = {
        '1': 'rbf', 
        '2': 'linear', 
        '3': 'poly'
    }
    kernel = kernels.get(kernel_choice, 'rbf')
    
    # Choix du paramètre C
    C = float(input("Entrez la valeur de C (défaut: 1.0): ") or 1.0)
    
    print(f"\nEntraînement du modèle SVM avec kernel {kernel} et C={C}...")
    model = SupportVectorMachine(kernel=kernel, C=C)
    model.fit(X_train, y_train)
    
    # Détails du modèle
    details = model.get_model_details()
    print("\nDétails du modèle SVM:")
    print(f"Kernel: {details['kernel']}")
    print(f"Paramètre C: {details['C']}")
    print(f"Nombre de vecteurs de support: {details['support_vectors_count']}")
    print(f"Classes: {details['classes']}")
    
    # Prédictions et métriques
    print("\nCalcul des prédictions et des métriques...")
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    
    # Affichage des résultats
    print("\nRésultats de la classification :")
    print(f"Matrice de confusion :")
    print(f"TP: {metrics['TP']}, TN: {metrics['TN']}")
    print(f"FP: {metrics['FP']}, FN: {metrics['FN']}")
    print(f"\nMétriques de performance :")
    print(f"Précision: {metrics['precision']:.3f}")
    print(f"Rappel: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    
    # Affichage de la matrice de confusion
    plot_confusion_matrix(metrics['confusion_matrix'], 
                        ['tested_negative', 'tested_positive'], 
                        f'SVM (kernel: {kernel})')
    visualize_svm_results(X_train, y_train, model, metrics)

def run_linear_regression_analysis():
    """Exécute l'analyse avec la régression linéaire."""
    file_path = input("Entrez le chemin du fichier ARFF (par défaut: 'diabetes.arff'): ") or '/home/ayoublb/Documents/MASTER 2/FD/TP/M2/diabetes.arff'
    print("\n=== Analyse avec Régression Linéaire ===")
    print("Chargement et préparation des données...")
    df, meta = load_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Configuration du modèle
    learning_rate = float(input("Taux d'apprentissage (défaut: 0.01): ") or 0.01)
    epochs = int(input("Nombre d'époques (défaut: 1000): ") or 1000)
    threshold = float(input("Seuil de classification (défaut: 0.5): ") or 0.5)
    
    print("\nEntraînement du modèle de régression linéaire...")
    model = LinearRegression(learning_rate=learning_rate, epochs=epochs, threshold=threshold)
    model.fit(X_train, y_train)
    
    # Détails du modèle
    details = model.get_model_details()
    print("\nDétails du modèle:")
    print(f"Learning rate: {details['learning_rate']}")
    print(f"Epochs: {details['epochs']}")
    print(f"Seuil: {details['threshold']}")
    print(f"Biais: {details['bias']:.4f}")
    print("\nPoids des features:")
    for idx, weight in details['feature_weights']:
        print(f"Feature {idx}: {weight:.4f}")
    print(f"\nPerte finale: {details['final_loss']:.4f}")
    
    # Prédictions et métriques
    predictions = model.predict(X_test)
    metrics = calculate_metrics(y_test, predictions)
    
    # Affichage des résultats
    print("\nRésultats de la classification:")
    print(f"Matrice de confusion:")
    print(f"TP: {metrics['TP']}, TN: {metrics['TN']}")
    print(f"FP: {metrics['FP']}, FN: {metrics['FN']}")
    print(f"\nMétriques de performance:")
    print(f"Précision: {metrics['precision']:.3f}")
    print(f"Rappel: {metrics['recall']:.3f}")
    print(f"F1-Score: {metrics['f1']:.3f}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    
    # Affichage de la matrice de confusion
    plot_confusion_matrix(metrics['confusion_matrix'], 
                        ['tested_negative', 'tested_positive'], 
                        'Linear Regression')
    
    print("\nAffichage des visualisations...")
    model.visualize_decision_boundary()

def run_apriori_analysis():
    """Exécute l'analyse avec l'algorithme Apriori."""
    # Chargement des données
    file_path = input("Entrez le chemin du fichier ARFF (par défaut: 'diabetes.arff'): ") or '/home/ayoublb/Documents/MASTER 2/FD/TP/M2/diabetes.arff'
    print("\n=== Analyse avec Apriori ===")
    print("Chargement et préparation des données...")
    df, meta = load_data(file_path)
    
    # Configuration des paramètres
    min_support = float(input("Support minimum (défaut: 0.3): ") or 0.3)
    min_confidence = float(input("Confiance minimum (défaut: 0.7): ") or 0.7)
    
    # Création et entraînement du modèle
    model = Apriori(min_support=min_support, min_confidence=min_confidence)
    print("\nAnalyse des patterns fréquents...")
    model.fit(df)
    
    # Affichage des résultats
    insights = model.get_insights()
    print(f"\nNombre total de règles trouvées: {insights['total_rules']}")
    
    print("\nTop 10 règles les plus pertinentes liées au diabète:")
    for i, rule in enumerate(insights['diabetes_rules'], 1):
        ant = ' ET '.join(rule['antecedent'])
        cons = ' ET '.join(rule['consequent'])
        print(f"\nRègle {i}:")
        print(f"SI {ant}")
        print(f"ALORS {cons}")
        print(f"Support: {rule['support']:.3f}")
        print(f"Confiance: {rule['confidence']:.3f}")
        print(f"Lift: {rule['lift']:.3f}")
    
    print("\nFacteurs de risque identifiés:")
    for factor in insights['risk_factors']:
        print(f"- {factor}")

def main():
    """Fonction principale."""
    while True:
        choice = print_menu()
        if choice == '1':
            run_knn_analysis()
        elif choice == '2':
            run_naive_bayes_analysis()
        elif choice == '3':
            run_decision_tree_analysis()
        elif choice == '4':  # Neural Network
                file_path = input("Entrez le chemin du fichier ARFF (par défaut: 'diabetes.arff'): ") or '/home/ayoublb/Documents/MASTER 2/FD/TP/M2/diabetes.arff'
                
                # Option de personnalisation
                print("\nConfiguration du réseau de neurones")
                hidden_layers_input = input("Entrez les couches cachées (séparées par des virgules, défaut: 16,8): ") or "16,8"
                hidden_layers = [int(layer) for layer in hidden_layers_input.split(',')]
                
                epochs_input = input("Nombre d'époques (défaut: 200): ") or "200"
                epochs = int(epochs_input)
                
                run_neural_network_analysis(file_path, hidden_layers=hidden_layers, epochs=epochs)
        elif choice == '5':  # Ajout de l'analyse SVM
            run_svm_analysis()
        elif choice == '6':
            run_linear_regression_analysis()
        elif choice == '7':
            run_apriori_analysis()
        elif choice == '8':
            print("Au revoir !")
            sys.exit(0)
        else:
            print("Option invalide. Veuillez réessayer.")
if __name__ == "__main__":
    main()
