import os
import sys
import argparse
from torch import nn
import torch.nn.functional as F
import torch
from dataset import BigEarthNetDataSet
from vit import ViT
    
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"{device=}")

    
def train_step(model, optimizer, image, label, criterion):
    optimizer.zero_grad()
    outputs = model(image)
    label = label.float()
    loss = criterion(outputs, label)  
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_multi_label(model, test_loader, device, writer=None, epoch=None, threshold=0.5):
    """
    Evaluate a multi-label classification model.
    
    Args:
        model: The PyTorch model to evaluate
        test_loader: DataLoader containing test data
        device: Device to run evaluation on
        writer: Optional TensorBoard writer
        epoch: Optional epoch number for logging
        threshold: Threshold for binary prediction (default: 0.5)
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    model.eval()
    total_samples = 0
    exact_matches = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    with torch.no_grad():
        for img, labels in test_loader:
            img = img.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            outputs = model(img)
            predictions = (torch.sigmoid(outputs) > threshold).float()
            
            # Calculate exact matches (all labels correct)
            exact_matches += (predictions == labels).all(dim=1).sum().item()
            
            # Calculate true positives, false positives, and false negatives
            true_positives = (predictions * labels).sum(dim=1)
            false_positives = (predictions * (1 - labels)).sum(dim=1)
            false_negatives = ((1 - predictions) * labels).sum(dim=1)
            
            total_true_positives += true_positives.sum().item()
            total_false_positives += false_positives.sum().item()
            total_false_negatives += false_negatives.sum().item()
            
            total_samples += img.size(0)

    # Calculate metrics
    metrics = {}
    metrics['exact_match_ratio'] = exact_matches / total_samples
    
    # Calculate micro-F1 score
    metrics['precision'] = total_true_positives / (total_true_positives + total_false_positives + 1e-10)
    metrics['recall'] = total_true_positives / (total_true_positives + total_false_negatives + 1e-10)
    metrics['micro_f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-10)

    # Log metrics to TensorBoard if writer is provided
    if writer is not None and epoch is not None:
        for metric_name, metric_value in metrics.items():
            writer.add_scalar(f"test/{metric_name}", metric_value, epoch)

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
    train_dataset = BigEarthNetDataSet('train', directory_path, file_path)
    test_dataset = BigEarthNetDataSet('test', directory_path, file_path)
    validation_dataset = BigEarthNetDataSet('validation', directory_path, file_path)

    batch_size = 256
    lr = 3e-4
    num_epochs = 15

    img_width = 120
    img_channels = 3
    num_classes = 19
    patch_size = 12
    embedding_dim = 128
    ff_dim = 512
    num_heads = 8 
    num_layers = 6
    weight_decay = 1e-4

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
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

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(f"runs/vit-big_earth_net_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")

    for epoch in range(num_epochs):
        losses = []
        # Initialize metrics for each epoch
        total_predictions = 0
        correct_predictions = 0
        
        model.train()
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            
            optimizer.zero_grad()
            outputs = model(img)  # Changed from 'image' to 'img' to match the loop variable
            label = label.float()
            
            # Calculate loss
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            # Calculate accuracy
            # Apply sigmoid to get predictions between 0 and 1
            predictions = torch.sigmoid(outputs)
            # Convert to binary predictions (0 or 1) using threshold of 0.5
            predicted_labels = (predictions > 0.5).float()
            
            # Calculate correct predictions
            # For multi-label, a prediction is "correct" if all labels match
            correct_predictions += (predicted_labels == label).all(dim=1).sum().item()
            total_predictions += label.size(0)
        
        # Calculate epoch metrics
        epoch_loss = sum(losses) / len(losses)
        epoch_accuracy = correct_predictions / total_predictions
        
        # Log metrics
        writer.add_scalar("train_loss", epoch_loss, epoch)
        writer.add_scalar("train_accuracy", epoch_accuracy, epoch)

        model.eval()
        metrics = evaluate_multi_label(
            model=model,
            test_loader=test_loader,
            device=device,
            writer=writer,
            epoch=epoch
        )

        print(f"{epoch=}")

if __name__ == "__main__":
    main()