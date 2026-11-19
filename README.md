# VisionOps

VisionOps est une plateforme expérimentale de Computer Vision orientée reproductibilité, serving et comparaison de performances.

## Current release: v0.1.0 foundation

The current stable release focuses on dataset preparation and validation. It currently provides:

- dataset registry for EuroSAT and Intel Image Classification;
- PyTorch/torchvision dataset loaders;
- explicit EuroSAT preparation command;
- ImageFolder-compatible data validation;
- JSON validation reports;
- automated tests for the data layer;
- an exploratory notebook for class distributions and sample images.

Training, MLflow tracking, model serving, Docker, CI, and hardware benchmarks are planned but are not implemented in v0.1.0.

## Roadmap

### v0.2.0 — reproducible training

- deterministic train/validation/test splits;
- train and evaluation transforms;
- ResNet-18 baseline;
- configuration-driven training;
- classification metrics and checkpoints.

### v0.3.0 — experiment tracking and model comparison

- MLflow tracking;
- error analysis;
- comparison of a lightweight CNN and ViT;
- reproducible benchmark reports.

### Future releases

- FastAPI inference service;
- Docker packaging and CI;
- ONNX export and CPU/GPU/Jetson benchmarks;
- cloud deployment and monitoring;
- detection, segmentation, tracking, image-to-text, and world-model extensions.

## Principes

- entraînement local sur la RTX 5070 Ti ;
- aucune ressource cloud persistante sans estimation de coût ;
- notebooks réservés à l'exploration et à l'analyse ;
- logique métier dans des modules Python testables ;
- résultats mesurés et reproductibles, jamais inventés.

## Status

The repository is currently in the dataset foundation stage. Only features explicitly listed as available above should be considered implemented.
