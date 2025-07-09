# Boîte à Outils d'Analyse de Machine Learning

Ce projet est une application de bureau complète, développée en Python avec Tkinter/PyQt5, conçue pour l'exploration de données et la comparaison de divers algorithmes de Machine Learning. Réalisé dans le cadre du module "Fouille de Données" (M1 Bio-Informatique), il fournit une plateforme interactive pour analyser des jeux de données, en particulier des données médicales comme le dataset *Pima Indians Diabetes*.

## 📸 Captures d'écran

![Interface Principale](docs/images/gui_main.png)
_Menu principal de l'application permettant de charger un fichier et de choisir entre classification supervisée et non supervisée._

![Fenêtre d'Analyse](docs/images/gui_results.png)
_Interface d'analyse où l'utilisateur peut sélectionner un algorithme et visualiser les résultats._

## ✨ Fonctionnalités

L'application offre un pipeline complet, du prétraitement des données à l'évaluation des modèles.

### 1. Analyse et Prétraitement de Données
- Chargement de fichiers de données (format ARFF).
- Analyse exploratoire : statistiques descriptives, visualisation des distributions (Boxplots, Scatter plots).
- Prétraitement : gestion des valeurs manquantes, normalisation des données (Min-Max, Z-score).

### 2. Algorithmes de Classification Supervisée
L'application implémente et compare les performances des algorithmes suivants :
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes Gaussien**
- **Arbre de Décision (avec Gain Ratio et Indice de Gini)**
- **Réseau de Neurones (MLP)**
- **Support Vector Machine (SVM)** avec différents noyaux (RBF, Linéaire, Polynomial).
- **Régression Linéaire** adaptée pour la classification.

### 3. Apprentissage Non Supervisé
- **Algorithme Apriori** pour la découverte de règles d'association et l'identification des facteurs de risque dans les données.

### 4. Évaluation de Modèles
- **Métriques complètes :** Précision, Rappel, F1-Score, et Accuracy.
- **Visualisation :** Matrice de confusion, courbes d'évolution des métriques (pour KNN), et visualisation de la frontière de décision (pour Régression Linéaire et SVM).

## 🛠️ Technologies et Bibliothèques

- **Python 3**
- **Tkinter / PyQt5** pour l'interface graphique.
- **Pandas** & **Numpy** pour la manipulation des données.
- **Scikit-learn** pour les modèles et les métriques.
- **Matplotlib** & **Seaborn** pour les visualisations.
- **SciPy** pour le chargement des fichiers ARFF.
- **PyDotPlus** & **Graphviz** pour la visualisation des arbres de décision.

## 🚀 Comment l'Exécuter

1.  **Clonez le dépôt :**
    ```bash
    git clone https://github.com/VOTRE_NOM_UTILISATEUR/ML-Analysis-Toolkit-GUI.git
    cd ML-Analysis-Toolkit-GUI
    ```
2.  **(Recommandé) Créez un environnement virtuel :**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
3.  **Installez les dépendances :**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Lancez l'application graphique :**
    ```bash
    python main_app.py 
    ```
    *(Remplacez `main_app.py` par le nom du fichier principal de votre GUI).*

## 📚 Documents de Référence
- **[Rapport Complet du Projet](./Rapport_Projet_ML.pdf)** : Analyse détaillée de chaque algorithme, résultats, et conclusions.