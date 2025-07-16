import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from generator.batch_generator import MultiClassBatchGenerator
from training.train_model import train_model
from testing.test_model import test_model
from data.io import save_as_pickle as save_data_pickle
import time
from copy import deepcopy
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

def run_model(model, batch_size, save_dir, best_model_name,
              train_generator_class, test_generator_class,
              train_generator_kwargs, test_generator_kwargs,
              epochs, device, optimizer, criterion, lr=1e-3,
              save_epoch=10, resume=True):

    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, best_model_name)
    metrics_path = os.path.splitext(model_path)[0] + ".pkl"

    # Initialize
    start_epoch = 1
    best_test_acc = 0
    best_epoch = 0
    best_model_state = None
    total_time_trained = 0.0

    if resume and os.path.exists(model_path) and os.path.exists(metrics_path):
        print(f"Resuming from: {model_path} and {metrics_path}")
        model.load_state_dict(torch.load(model_path))

        with open(metrics_path, 'rb') as f:
            metrics = pickle.load(f)

        # Convert arrays back to lists for appending
        for key in ['train_loss', 'test_loss', 'train_acc', 'test_acc']:
            if isinstance(metrics.get(key), np.ndarray):
                metrics[key] = metrics[key].tolist()

        start_epoch = len(metrics['train_loss']) + 1

        if len(metrics['test_acc']) > 0:
            best_test_acc = float(np.max(metrics['test_acc']))
            best_epoch = int(np.argmax(metrics['test_acc'])) + 1
        else:
            best_test_acc = 0.0
            best_epoch = 0

        total_time_trained = metrics.get('total_time', 0.0)

        print(f"Resumed from epoch {start_epoch - 1}, previous best acc: {best_test_acc:.4f} at epoch {best_epoch}")
    else:
        metrics = {
            'train_loss': [],
            'test_loss': [],
            'train_acc': [],
            'test_acc': [],
            'best_epoch': 0,
            'best_test_acc': 0.0,
            'total_time': 0.0,
            'time_per_epoch': 0.0
        }

    # Start epoch timer
    epoch_start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, epochs + 1):
        train_loader = DataLoader(
            train_generator_class(**train_generator_kwargs),
            batch_size=batch_size,
            collate_fn=MultiClassBatchGenerator.collate_data
        )
        test_loader = DataLoader(
            test_generator_class(**test_generator_kwargs),
            batch_size=batch_size,
            collate_fn=MultiClassBatchGenerator.collate_data
        )

        train_results = train_model(model, train_loader, optimizer, criterion, device)
        test_results = test_model(model, test_loader, criterion, device)

        metrics['train_loss'].append(train_results['loss'])
        metrics['test_loss'].append(test_results['loss'])
        metrics['train_acc'].append(train_results['acc'])
        metrics['test_acc'].append(test_results['acc'])

        if test_results['acc'] > best_test_acc:
            best_test_acc = test_results['acc']
            best_epoch = epoch
            best_model_state = deepcopy(model.state_dict())
            metrics['best_epoch'] = best_epoch
            metrics['best_test_acc'] = best_test_acc

        if epoch % save_epoch == 0 or epoch == epochs:
            save_metrics = {
                'train_loss': np.array(metrics['train_loss']),
                'test_loss': np.array(metrics['test_loss']),
                'train_acc': np.array(metrics['train_acc']),
                'test_acc': np.array(metrics['test_acc']),
                'best_epoch': metrics['best_epoch'],
                'best_test_acc': metrics['best_test_acc'],
                'total_time': metrics['total_time'],
                'time_per_epoch': metrics['time_per_epoch']
            }
            torch.save(model.state_dict(), model_path)
            save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), save_metrics)
            print(f"Saved model and metrics at epoch {epoch} to {model_path}")

        print(f"Epoch: {epoch:03d}/{epochs} | "
              f"Train Loss: {train_results['loss']:.4f} | Train Acc: {train_results['acc']:.4f} | "
              f"Test Loss: {test_results['loss']:.4f} | Test Acc: {test_results['acc']:.4f} | "
              f"Best Test Acc: {best_test_acc:.4f}")

    # End epoch timer
    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - epoch_start_time
    metrics['total_time'] = total_time_trained + elapsed_time
    metrics['time_per_epoch'] = metrics['total_time'] / len(metrics['train_loss'])

    total_min, total_sec = divmod(metrics['total_time'], 60)
    print(f"\nTotal training time: {int(total_min)} min {total_sec:.2f} sec")
    print(f"Avg time per epoch: {metrics['time_per_epoch']:.2f} sec")

    # Final inference
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        torch.save(best_model_state, model_path)
        print(f"Final model saved from epoch {best_epoch} with test accuracy: {best_test_acc:.4f}")
    else:
        print("Warning: No best model found, using final model weights")

    final_train_loader = DataLoader(
        train_generator_class(**train_generator_kwargs),
        batch_size=batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data
    )
    final_test_loader = DataLoader(
        test_generator_class(**test_generator_kwargs),
        batch_size=batch_size,
        collate_fn=MultiClassBatchGenerator.collate_data
    )

    metrics['final_train'] = run_inference(model, final_train_loader, device)
    metrics['final_test'] = run_inference(model, final_test_loader, device)

    from sklearn.metrics import classification_report
    metrics['classification_report'] = classification_report(
        metrics['final_test']['labels'],
        metrics['final_test']['preds'],
        output_dict=True
    )

    final_metrics = {
        'train_loss': np.array(metrics['train_loss']),
        'test_loss': np.array(metrics['test_loss']),
        'train_acc': np.array(metrics['train_acc']),
        'test_acc': np.array(metrics['test_acc']),
        'best_epoch': metrics['best_epoch'],
        'best_test_acc': metrics['best_test_acc'],
        'final_train_preds': metrics['final_train']['preds'],
        'final_train_scores': metrics['final_train']['scores'],
        'final_train_labels': metrics['final_train']['labels'],
        'final_test_preds': metrics['final_test']['preds'],
        'final_test_scores': metrics['final_test']['scores'],
        'final_test_labels': metrics['final_test']['labels'],
        'classification_report': metrics['classification_report'],
        'total_time': metrics['total_time'],
        'time_per_epoch': metrics['time_per_epoch']
    }

    save_data_pickle(os.path.basename(metrics_path), os.path.dirname(metrics_path), final_metrics)

    print(f"\nTraining complete. Best model at epoch {best_epoch} with test accuracy: {best_test_acc:.4f}")
    print(f"Final results saved to {metrics_path}")

    return final_metrics, model