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
from transformers import get_cosine_schedule_with_warmup
    
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
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    model.eval()
    total_samples = 0
    total_loss = 0.0
    exact_matches = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    
    with torch.no_grad():
        for img, labels in loader:
            img = img.to(device)
            labels = labels.to(device)
            
            outputs = model(img)
            # Calculate loss
            batch_loss = criterion(outputs, labels)
            total_loss += batch_loss.item()
            
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
    
    # Calculate average loss
    avg_loss = total_loss / total_samples
    
    metrics = {}
    metrics['loss'] = avg_loss
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
    parser.add_argument('--checkpoint', help='Path to checkpoint file to resume training from', default=None)
    
    args = parser.parse_args()

    train_dataset = BigEarthNetDataSet('train')
    test_dataset = BigEarthNetDataSet('test')
    validation_dataset = BigEarthNetDataSet('validation')

    batch_size = 256
    lr = 4e-3
    num_epochs = 40 

    # Default hyperparameters
    img_width = 120
    img_channels = 5
    num_classes = 19
    patch_size = 8
    embedding_dim = 256
    ff_dim = 1536
    num_heads = 8 
    num_layers = 6 
    weight_decay = 2e-4

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

    start_epoch = 0
    best_val_f1 = 0
    optimal_thresholds = torch.ones(num_classes).to(device) * 0.5

    # Load checkpoint if provided
    if args.checkpoint and os.path.isfile(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Update hyperparameters from checkpoint
        hp = checkpoint['hyperparameters']
        img_width = hp['img_width']
        img_channels = hp['img_channels']
        patch_size = hp['patch_size']
        embedding_dim = hp['embedding_dim']
        ff_dim = hp['ff_dim']
        num_heads = hp['num_heads']
        num_layers = hp['num_layers']
        num_classes = hp['num_classes']
        
        # Create model with loaded hyperparameters
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
        
        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Initialize optimizer after model parameters are loaded
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        start_epoch = checkpoint['epoch'] + 1
        best_val_f1 = checkpoint['best_val_f1']
        optimal_thresholds = checkpoint['thresholds']
        
        print(f"Resuming from epoch {start_epoch} with best F1 score: {best_val_f1:.4f}")
    else:
        # Create new model if no checkpoint
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    count_parameters(model)

    steps_per_epoch = len(train_loader)
    num_training_steps = num_epochs * steps_per_epoch
    num_warmup_steps = num_training_steps * 0.05 

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    writer = SummaryWriter(f"runs/vit-big_earth_net_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}")
    early_stopping = EarlyStopping(patience=6, verbose=True)

    for epoch in range(start_epoch, num_epochs):
        print(f"Starting epoch: {epoch}")

        losses = []
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

        early_stopping(metrics['loss'], model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

        scheduler.step()
    
        if metrics['micro_f1'] > best_val_f1:
            best_val_f1 = metrics['micro_f1']
            best_thresholds = optimal_thresholds.clone()

            # Save the model and thresholds
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_f1': best_val_f1,
                'epoch': epoch,
                'thresholds': best_thresholds,
                'hyperparameters': {
                    'img_width': img_width,
                    'img_channels': img_channels,
                    'patch_size': patch_size,
                    'embedding_dim': embedding_dim,
                    'ff_dim': ff_dim,
                    'num_heads': num_heads,
                    'num_layers': num_layers,
                    'num_classes': num_classes
                }
            }
            torch.save(checkpoint, f'model_checkpoint_f1_{best_val_f1:.3f}.pt')

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