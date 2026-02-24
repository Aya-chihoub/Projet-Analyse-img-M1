# Detection et Identification de Pieces en Euro

**Master 1 Informatique — IFLBX030 Introduction a l'analyse d'images**

## Objectif

Detecter et identifier les pieces en euro dans une image afin de :
- Compter le nombre de pieces
- Estimer la valeur monetaire totale

Le projet utilise des techniques classiques de traitement d'images avec un minimum d'apprentissage automatique (KNN), sans deep learning.

## Structure du projet

```
coin_detection.py        # Detection : localisation des pieces
coin_identification.py   # Identification : classification et denomination
coin_detector.py         # Orchestrateur : pipeline complet
main.py                  # Script principal (execution + evaluation)
ground_truth.csv         # Verite terrain (106 images)
requirements.txt         # Dependances Python
data/                    # Images a analyser
results/                 # Resultats generes

# Notebooks d'accompagnement et de justification des choix
notebook_Detection.ipynb       # Détail des étapes de segmentation et détection
notebook_Identification.ipynb  # Détail des analyses colorimétriques (LAB/HSV)
notebook_Evaluation.ipynb      # Analyse statistique approfondie des résultats
```

## Utilisation

```bash
pip install -r requirements.txt

# Traiter toutes les images + evaluation
python main.py

# Traiter une seule image
python main.py data/exemple1.jpg
```

## Pipeline

### 1. Preprocessing
- Redimensionnement (largeur cible = 800px, hauteur max = 1000px)
- Amelioration du contraste via CLAHE
- Flou gaussien

### 2. Analyse du fond
- Echantillonnage des pixels de bordure en HSV
- Classification du fond : clair/sombre et colore/neutre

### 3. Segmentation
- **Fond neutre** : seuillage d'Otsu sur l'image en niveaux de gris + operations morphologiques
- **Fond colore** : seuillage d'Otsu sur le canal de saturation (les pieces metalliques ont une saturation bien plus faible que les surfaces colorees)

### 4. Detection
Trois methodes combinees :
- **Hough Circle Transform** : detection de cercles avec balayage multi-parametres
- **Watershed** : separation des pieces qui se touchent via la transformee de distance
- **Contours** : detection par analyse de contours pour les pieces bien separees

Les resultats sont fusionnes (strategie "le plus de pieces gagne"), puis filtres par NMS (suppression des doublons) et coherence des rayons.

### 5. Classification des couleurs
- Extraction de features : moyenne HSV, moyenne Lab (a\*, b\*), ecart-type de saturation, score bimetallique ameliore
- Classification par regles adaptatives selon le type de fond
- Raffinement optionnel par KNN (k=5, 30 exemples d'entrainement)
- Trois groupes : **cuivre** (1c, 2c, 5c), **or** (10c, 20c, 50c), **bimetallique** (1 euro, 2 euro)

### 6. Denomination
Attribution basee sur la taille relative au sein de chaque groupe de couleur, en utilisant les diametres reels des pieces en euro comme reference.

## Resultats

**Detection (106 images)**

| Metrique | Valeur |
|---|---|
| Precision | 97.48% |
| Rappel | 84.01% |
| F1-Score | 90.25% |
| Pieces reelles | 1107 |
| Pieces detectees | 954 |
| Comptage exact | 56.60% |

**Identification / Evaluation financiere**

| Metrique | Valeur |
|---|---|
| MAE financiere | 1.90 EUR par image |
| Exactitude financiere globale | 6.60% |
| Temps moyen par image | 0.24s |

**Par groupe**

| Groupe | Images | MAE (EUR) | Comptage exact |
|---|---|---|---|
| gp1 | 14 | 0.62 | 64.29% |
| gp2 | 15 | 0.92 | 40.00% |
| gp3 | 10 | 5.59 | 60.00% |
| gp4 | 10 | 0.65 | 70.00% |
| gp5 | 25 | 2.99 | 32.00% |
| gp6 | 10 | 1.57 | 90.00% |
| gp7 | 12 | 0.58 | 83.33% |
| gp8 | 10 | 1.89 | 50.00% |

## Difficultes rencontrees

- **Fonds colores (rouge, vert)** : la segmentation classique en niveaux de gris echouait completement. Solution : basculer sur le canal de saturation HSV, ou le contraste metal/surface coloree est net.

- **Pieces qui se touchent** : la detection de Hough seule ne suffisait pas. L'ajout de l'algorithme Watershed a permis de separer les groupes de pieces.

- **Faux positifs / faux negatifs** : equilibrer les parametres de Hough (strict a relaxe) et la validation par contraste a necessite de nombreuses iterations. Un jeu de 5 parametres a ete conserve pour couvrir un maximum de cas.

- **Performance** : les premieres versions prenaient plus de 60s par image. L'optimisation principale a ete de pre-calculer les conversions d'image (gris, HSV) une seule fois et de les reutiliser dans la validation, ramenant le temps moyen a ~0.24s.

- **Regressions** : chaque amelioration pour un type d'image risquait de degrader les resultats sur d'autres. Une evaluation systematique sur les 106 images apres chaque modification a ete indispensable.

## Dependances

- OpenCV (`opencv-python`)
- NumPy
- scikit-learn (KNN uniquement)
- Matplotlib (graphiques d'evaluation)
- Seaborn
- Pandas