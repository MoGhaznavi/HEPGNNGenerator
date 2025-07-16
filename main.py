import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from data.io import load_pickle_file
from generator.batch_generator import MultiClassBatchGenerator
from model.edge_classifier import MultiEdgeClassifier
from training.train_model import train_model
from testing.test_model import test_model
from experiment.run_model import run_model
from config.config import config
from utils.misc import compute_sqrt_inverse_weights, run_inference

def main():
    # Load data
    load_path = config['data_path']
    scaled_data = load_pickle_file("scaled_data.pkl", load_path)
    neighbor_pairs_list = load_pickle_file("neighbor_pairs_list.pkl", load_path)
    labels_for_neighbor_pairs = load_pickle_file("labels_for_neighbor_pairs.pkl", load_path)


    # Compute class weights
    weight_tensor = compute_sqrt_inverse_weights(
        labels=labels_for_neighbor_pairs,
        num_classes=config['num_classes'],
        device=config['device']
    )

    # Initialize model
    model = MultiEdgeClassifier(
        input_dim=config['num_features'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['num_classes'],
        device=config['device'],
        num_layers=config['num_layers']
    ).to(config['device'])

    optimizer = optim.Adam(
        model.parameters(), lr=config['lr'], weight_decay=config['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Generator kwargs
    gen_kwargs = {
        'features_dict': scaled_data,
        'neighbor_pairs': neighbor_pairs_list,
        'labels': labels_for_neighbor_pairs,
        'class_counts': config['class_counts'],
        'batch_size': config['num_events_per_batch']
    }

    # Train and evaluate
    metrics, model = run_model(
        model=model,
        batch_size=config['num_events_per_batch'],
        save_dir=config['save_dir'],
        best_model_name=config['model_name'],
        train_generator_class=MultiClassBatchGenerator,
        test_generator_class=MultiClassBatchGenerator,
        train_generator_kwargs={**gen_kwargs, 'mode': 'train'},
        test_generator_kwargs={**gen_kwargs, 'mode': 'test'},
        epochs=config['epochs'],
        device=config['device'],
        optimizer=optimizer,
        criterion=criterion,
        lr=config['lr']
        save_epoch=config['save_epoch'],
        resume=config[resume]
    )
    
    # Print timing summary from metrics
    total_time = metrics['total_time']
    time_per_epoch = metrics['time_per_epoch']
    min_total, sec_total = divmod(total_time, 60)
    print(f"\nTotal Training Time: {int(min_total)} min {sec_total:.2f} sec")
    print(f"Avg Time per Epoch: {time_per_epoch:.2f} sec")
    print(f"Best model saved to: {os.path.join(config['save_dir'], config['model_name'])}")

if __name__ == "__main__":
    main()