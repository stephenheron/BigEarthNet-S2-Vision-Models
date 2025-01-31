import os
import sys
import argparse
from torch import nn
import torch.nn.functional as F
import torch
from dataset import BigEarthNetDataSet
from vit import ViT
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from early_stopping_pytorch import EarlyStopping
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"{device=}")

def find_optimal_thresholds(model, loader, device):
    model.eval()
    # Initialize lists to store all predictions and labels
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for img, labels in loader:
            img = img.to(device)
            labels = labels.to(device)
            outputs = model(img)
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions)
            all_labels.append(labels)
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    num_labels = all_predictions.shape[1]
    optimal_thresholds = []
    
    # Find optimal threshold for each label
    for label_idx in range(num_labels):
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in torch.linspace(0.1, 0.9, 100):
            pred = (all_predictions[:, label_idx] > threshold).float()
            true = all_labels[:, label_idx]
            
            # Calculate F1 score components
            true_positives = (pred * true).sum()
            false_positives = (pred * (1 - true)).sum()
            false_negatives = ((1 - pred) * true).sum()
            
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                
        optimal_thresholds.append(best_threshold)
    
    return torch.tensor(optimal_thresholds).to(device)

def count_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'{name}: {param.numel():,} parameters')
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total: {total}')
    
def evaluate_multi_label(model, loader, type, device, writer=None, epoch=None, thresholds=None):
    if thresholds is None:
        thresholds = torch.ones(model.num_classes).to(device) * 0.5
        
    model.eval()
    total_samples = 0
    exact_matches = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    with torch.no_grad():
        for img, labels in loader:
            img = img.to(device)
            labels = labels.to(device)
            
            outputs = model(img)
            # Use per-label thresholds
            predictions = (torch.sigmoid(outputs) > thresholds[None, :]).float()
            
            exact_matches += (predictions == labels).all(dim=1).sum().item()
            
            true_positives = (predictions * labels).sum(dim=1)
            false_positives = (predictions * (1 - labels)).sum(dim=1)
            false_negatives = ((1 - predictions) * labels).sum(dim=1)
            
            total_true_positives += true_positives.sum().item()
            total_false_positives += false_positives.sum().item()
            total_false_negatives += false_negatives.sum().item()
            
            total_samples += img.size(0)

    metrics = {}
    metrics['exact_match_ratio'] = exact_matches / total_samples
    metrics['precision'] = total_true_positives / (total_true_positives + total_false_positives + 1e-10)
    metrics['recall'] = total_true_positives / (total_true_positives + total_false_negatives + 1e-10)
    metrics['micro_f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-10)

    if writer is not None and epoch is not None:
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f"{type}/{metric_name}", metric_value, epoch)

    return metrics

def main():
    parser = argparse.ArgumentParser(description='Process a specific file in a directory')
    parser.add_argument('directory', help='Path to the BigEarthNet directory')
    parser.add_argument('filename', help='Name of the metadata parquet file to process')
    
    args = parser.parse_args()

    directory_path = args.directory
    filename = args.filename
    file_path = os.path.join(directory_path, filename)
    
    # Verify the directory exists
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"'{directory_path}' is not a valid directory")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"'{filename}' not found in '{directory_path}'")

    #possible splits: ['test' 'validation' 'train']
    train_dataset = BigEarthNetDataSet('train')
    test_dataset = BigEarthNetDataSet('test')
    validation_dataset = BigEarthNetDataSet('validation')

    batch_size = 256
    lr = 5e-4
    num_epochs = 32

    img_width = 120
    img_channels = 3
    num_classes = 19
    patch_size = 8
    embedding_dim = 512
    ff_dim = 1024
    num_heads = 8 
    num_layers = 6 
    weight_decay = 1e-3

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    from torch.utils.tensorboard import SummaryWriter
    from datetime import datetime

    model = ViT(
        img_width=img_width,
        img_channels=img_channels,
        patch_size=patch_size,
        d_model=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        num_classes=num_classes,
        ff_dim=ff_dim,
    ).to(device)

    count_parameters(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True, min_lr=1e-6)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_val_f1 = 0
    optimal_thresholds = torch.ones(num_classes).to(device) * 0.5

    writer = SummaryWriter(f"runs/vit-big_earth_net_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    early_stopping = EarlyStopping(patience=4, verbose=True)

    for epoch in range(num_epochs):
        print(f"Starting epoch: {epoch}")

        losses = []
        # Initialize metrics for each epoch
        total_predictions = 0
        correct_predictions = 0
        
        model.train()
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(img) 
                label = label.float()
                loss = criterion(outputs, label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(loss.item())
            epoch_loss = sum(losses) / len(losses)
            writer.add_scalar("train_loss", epoch_loss, epoch)

        model.eval()
        optimal_thresholds = find_optimal_thresholds(model, validation_loader, device)
        print("Optimal thresholds:", optimal_thresholds)

        metrics = evaluate_multi_label(
            model=model,
            loader=validation_loader,
            type="validation",
            device=device,
            writer=writer,
            epoch=epoch,
            thresholds=optimal_thresholds
        )

        scheduler.step(metrics['micro_f1'])
    
        if metrics['micro_f1'] > best_val_f1:
            best_val_f1 = metrics['micro_f1']
            best_thresholds = optimal_thresholds.clone()

    model.eval()
    metrics = evaluate_multi_label(
        model=model,
        loader=test_loader,
        type="test",
        device=device,
        writer=writer,
        epoch=epoch,
        thresholds=optimal_thresholds
    )

if __name__ == "__main__":
    main()