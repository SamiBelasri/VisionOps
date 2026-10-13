# VisionOps

VisionOps est une plateforme expérimentale de Computer Vision orientée reproductibilité, serving et comparaison de performances.

## VisionOps v0.1

Le premier périmètre couvre :

- classification d'images sur EuroSAT et Intel Image Classification ;
- comparaison d'un CNN léger et d'un ViT léger ;
- entraînement reproductible et suivi MLflow ;
- API FastAPI conteneurisée ;
- tests automatisés et CI ;
- benchmarks CPU/GPU.

Les extensions détection, segmentation, tracking, image-to-text et world models sont prévues, mais hors périmètre de v0.1.

## Principes

- entraînement local sur la RTX 5070 Ti ;
- aucune ressource cloud persistante sans estimation de coût ;
- notebooks réservés à l'exploration et à l'analyse ;
- logique métier dans des modules Python testables ;
- résultats mesurés et reproductibles, jamais inventés.

## État

Phase 0 — cahier des charges et squelette du dépôt.
