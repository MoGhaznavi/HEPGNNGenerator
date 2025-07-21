# Edge Classification Pipeline (Used in HEP Setting)
A modular and scalable PyTorch-based framework for edge classification on graph-structured data, with support for custom batch generation, training, testing, inference, and metric visualization.

## Project structure
The pipeline has the following directory structure

Config Folder
- config.py
    - Global parameters and runtime settings
- plot_config.py
    - Plotting configurations

Data Folder
- io.py
    - Functions for loading and saving .pkl files

Experiment Folder
- run_model.py
    - File for training, evaluation, and inference

Generator Folder
- batch_generator.py
    - Batch generation for edge pairs

Metrics Folder
- analysis.py
    - Performance metrics, confusion matrices, and ROC/TPR/FPR analysis

Model Folder
- edge_classifier.py
    - GNN-based edge classification model

Testing Folder
- test_model.py
    - Testing pipeline and evaluation metrics

Training Folder
- train_model.py
    - Training loop with checkpointing and AMP support

Utils Folder
- misc.py
    - Helper functions: weighted loss computation, inference logic

main.py - Top-level script: integrates training, testing, and inference

plot_metrics.py - Script to visualize performance metrics
