# Détection et Identification de Pièces de Monnaie en Euros

**Master 1 Informatique - IFLBX030 Introduction à l'analyse d'images**  
Université Paris Cité

## Objectif

Concevoir un programme permettant, à partir d'une image donnée en entrée, de compter le nombre de pièces et d'estimer la somme (en euros) représentée.

## Structure du Projet

```
Analuyse_d_images_projet/
├── data/                    # Images de pièces (10 images)
│   ├── exemple1.jpg
│   ├── exemple2.jpg
│   └── ...
├── src/
│   ├── detection.py         # Partner A - Détection des pièces
│   ├── identification.py    # Partner B - Identification et comptage
│   └── main.py              # Pipeline principal
├── results/                 # Résultats et visualisations
├── ground_truth.json        # Vérité terrain
├── requirements.txt         # Dépendances Python
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

```bash
cd src

# Traiter toutes les images et évaluer
python main.py

# Traiter une seule image
python main.py ../data/exemple1.jpg
```

## Répartition des Tâches

### Partner A - Détection (`detection.py`)
- Chargement et prétraitement de l'image
- Détection des cercles (Hough Transform)
- Extraction des régions de pièces
- Visualisation

### Partner B - Identification (`identification.py`)
- Analyse de la couleur (cuivre/or/bimétallique)
- Classification par taille et couleur
- Comptage et calcul de la valeur totale

## Méthode

1. **Prétraitement**: Conversion en niveaux de gris, flou gaussien
2. **Détection**: Transformation de Hough pour détecter les cercles
3. **Identification**: Classification basée sur le rayon (taille) et la couleur
4. **Évaluation**: Comparaison avec la vérité terrain

## Évaluation

Les métriques calculées :
- Taux de détection (coins detected / coins actual)
- Erreur de comptage
- Erreur de valeur (en euros)
