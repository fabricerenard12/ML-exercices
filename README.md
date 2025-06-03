# Formation Machine Learning

Bienvenue dans ce dépôt d’exercices pour la formation en Machine Learning !  
Cette formation a pour but d’explorer les concepts fondamentaux du machine learning à travers des implémentations pratiques en Python et l’utilisation de bibliothèques populaires comme `scikit-learn`, `numpy`, `pytorch`, etc.

## Contenu

Chaque répertoire ou fichier aborde un thème ou un algorithme particulier du ML. Voici un aperçu des sujets couverts :

- Régression linéaire
- Régression polynomiale
- Régression logistique
- Arbres de décision, Forêts aléatoires et Gradient Boosting
- Réseaux de neurones

## Structure typique d’un exercice

Chaque exercice suit généralement la structure suivante :

1. **Séparation en jeu d’entraînement / validation**
2. **Implémentation et entraînement du modèle**
3. **Évaluation du modèle avec des métriques pertinentes**

## Pré-requis

Avant de commencer, assurez‑vous d’avoir :
- **Suivi le MOOC** : [Initiez‑vous au Machine Learning](https://openclassrooms.com/fr/courses/8063076-initiez-vous-au-machine-learning) (OpenClassrooms)
- **Lu les chapitres 1 à 4** de la série [Neural Networks](https://www.3blue1brown.com/topics/neural-networks) (3Blue1Brown)
- **Réalisé le tutoriel** : [PyTorch Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
- **Python 3.8 +** installé
- **Dépendances du projet** installées :

```bash
pip install -r requirements.txt
```

## Exécution des exercices

Chaque dossier possède un fichier `main.py` que vous pouvez exécuter directement :

```bash
python src/linear_regression/main.py      # exemple
```

> Remplacez le chemin par celui de l'exercice que vous souhaitez lancer.

## Lancement des tests

Les tests unitaires sont écrits avec **pytest**. Depuis la racine du dépôt :

```bash
pytest -q
```

Vous pouvez également exécuter un seul fichier de tests :

```bash
pytest tests/test_linear_regression.py      # exemple
```
