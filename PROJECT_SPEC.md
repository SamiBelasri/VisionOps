# VisionOps v0.1 — Cahier des charges

## Objectif

Démontrer un cycle complet de modèle de classification d'images : données, entraînement, évaluation, tracking, export, API et benchmark.

## Critères de réussite

1. Un environnement vierge peut installer le projet.
2. EuroSAT et Intel Image Classification utilisent une interface de dataset commune.
3. Un entraînement est relançable depuis une configuration versionnée.
4. Les paramètres, métriques et artefacts sont enregistrés dans MLflow.
5. Une API retourne classe, confiance, version du modèle et latence.
6. Docker et les tests fonctionnent localement.
7. Le README documente les limites et les performances réellement mesurées.

## Hors périmètre initial

Kubernetes, entraînement GPU cloud, multi-cloud, fine-tuning de grands modèles multimodaux et world models.

## Risques

- datasets difficiles à télécharger ou à redistribuer ;
- comparaison abusive des accuracy entre datasets ;
- dépendances trop lourdes ;
- dérive vers une plateforme générique avant d'avoir un pipeline fonctionnel.
