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
ground_truth.csv         # Verite terrain (37 images)
requirements.txt         # Dependances Python
data/                    # Images a analyser
results/                 # Resultats generes
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

| Metrique | Global | gp4 (22 images) | gp2 (15 images) |
|---|---|---|---|
| MAE comptage | 1.05 | 0.50 | 1.87 |
| Comptage exact | 59.5% | 77.3% | 33.3% |
| MAE valeur (EUR) | 0.96 | 0.69 | 1.37 |
| Valeur a +/-0.50 EUR | 44.4% | 50.0% | 35.7% |

## Difficultes rencontrees

- **Fonds colores (rouge, vert)** : la segmentation classique en niveaux de gris echouait completement. Solution : basculer sur le canal de saturation HSV, ou le contraste metal/surface coloree est net.

- **Pieces qui se touchent** : la detection de Hough seule ne suffisait pas. L'ajout de l'algorithme Watershed a permis de separer les groupes de pieces.

- **Faux positifs / faux negatifs** : equilibrer les parametres de Hough (strict a relaxe) et la validation par contraste a necessite de nombreuses iterations. Un jeu de 5 parametres a ete conserve pour couvrir un maximum de cas.

- **Performance** : les premieres versions prenaient plus de 60s par image. L'optimisation principale a ete de pre-calculer les conversions d'image (gris, HSV) une seule fois et de les reutiliser dans la validation, ramenant le temps moyen a ~3.5s.

- **Regressions** : chaque amelioration pour un type d'image risquait de degrader les resultats sur d'autres. Une evaluation systematique sur les 37 images apres chaque modification a ete indispensable.

## Dependances

- OpenCV (`opencv-python`)
- NumPy
- scikit-learn (KNN uniquement)
- Matplotlib (graphiques d'evaluation)
