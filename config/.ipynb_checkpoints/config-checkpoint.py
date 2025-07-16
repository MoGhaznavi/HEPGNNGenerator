import torch

config = {
    'data_path': "/storage/mxg1065/datafiles",
    'save_dir': "/storage/mxg1065/baseline_models",
    'model_name': "default_file_name.pt",
    'num_events_per_batch': 1,
    'num_features': 5,
    'num_classes': 5,
    'hidden_dim': 128,
    'num_layers': 12
    'epochs': 200,
    'class_counts': {0: 40000, 1: 40000, 2: 40000, 3: 40000, 4: 3000},
    'lr': 1e-3,
    'weight_decay': 5e-4,
    'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    'save_epoch': 5,
    'resume': True
}
