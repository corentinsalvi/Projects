# Jeu d'échecs 3D

![Plateau 3D — capture d'écran](statique/étiquette.png)


## Description
Jeu d'échecs en 3D permettant de jouer contre le moteur Stockfish (MinMax). Le projet combine la logique stratégique du jeu et une expérience visuelle fluide : rotation/zoom de la caméra, animations des pièces et rendu 3D en temps réel.

## Fonctionnalités principales
- Affichage 3D du plateau et des pièces avec contrôle caméra (rotation, zoom, déplacement).
- Détection et validation des coups selon les règles officielles.
- Gestion complète de la partie : coups, prises, échec, échec et mat, promotion.
- Intégration de Stockfish pour l'IA.
- Animations pour les déplacements et rendu 3D temps réel.

## Technologies
- Python : logique du jeu et intégration de Stockfish.
- JavaScript : modélisation et rendu 3D.
- HTML / CSS : interface utilisateur.

## Prérequis
- Python 3.8+ installé
- Exécutable Stockfish (mettez le chemin dans la config ou dans le dossier du projet)
- Navigateur moderne pour la partie front-end

## Installation et exécution
1. Cloner le dépôt :
    ```bash
    git clone <url-du-repo>
    cd <nom-du-repo>
    ```
2. Créer et activer un environnement virtuel :
    - Windows :
      ```powershell
      python -m venv EnvVirtuel
      .\EnvVirtuel\Scripts\Activate.ps1
      ```
3. Installer les dépendances :
    ```bash
    pip install -r bibliotheques-necessaires.txt
    ```
4. Lancer l'application :
    ```bash
    python main.py
    ```
6. Ouvrir l'URL locale fournie dans le terminal (ex. http://localhost:5000) et jouer.

## Utilisation
- Utiliser la souris / les contrôles affichés pour tourner, zoomer et déplacer la caméra.
- Cliquer une pièce puis sa destination pour jouer un coup.
- L'IA répond automatiquement via Stockfish.

## Contribuer
- Ouvrir une issue pour signaler bugs ou proposer améliorations.
- Faire une branche par fonctionnalité puis un pull request clair et documenté.

## Licence
Voir le fichier LICENSE du projet.

Bon développement et bon jeu !
