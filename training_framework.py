import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from early_stopping_pytorch import EarlyStopping
from datetime import datetime
import os
from tqdm import tqdm

class BaseModel(nn.Module):
    """Base class for all models"""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def get_parameters(self):
        """Return model parameters for printing"""
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total_parameters': total,
            'model_type': self.__class__.__name__
        }
    
class ModelTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        learning_rate=1e-3,
        weight_decay=1e-4,
        num_epochs=80,
        patience=6,
        checkpoint=None  # Add checkpoint parameter
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Initialize or restore from checkpoint
        if checkpoint is not None:
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_f1 = checkpoint['best_val_f1']
            
            # Restore optimizer state
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore thresholds if they exist
            self.optimal_thresholds = checkpoint.get('thresholds', 
                torch.ones(self.model.num_classes).to(self.device) * 0.5)
        else:
            self.start_epoch = 0
            self.best_val_f1 = 0
            self.optimal_thresholds = torch.ones(self.model.num_classes).to(self.device) * 0.5
            
            # Create new optimizer
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.scaler = GradScaler()
        
        # Setup tensorboard with continued run if resuming
        run_name = f"{model.__class__.__name__}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
        if checkpoint is not None:
            run_name += "_resumed"
        self.writer = SummaryWriter(f"runs/{run_name}")
        
        # Setup early stopping
        self.early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        # Setup learning rate scheduler
        steps_per_epoch = len(train_loader)
        self.scheduler = model.get_learning_rate_scheduler(
            self.optimizer, 
            steps_per_epoch, 
            num_epochs - self.start_epoch  # Adjust for remaining epochs
        )

    def find_optimal_thresholds(self):
        """Find optimal thresholds for multi-label classification"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for img, labels in self.val_loader:
                img = img.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(img)
                predictions = torch.sigmoid(outputs)
                all_predictions.append(predictions)
                all_labels.append(labels)
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        optimal_thresholds = []
        for label_idx in range(self.model.num_classes):
            best_f1 = 0
            best_threshold = 0.5
            
            for threshold in torch.linspace(0.1, 0.9, 100):
                pred = (all_predictions[:, label_idx] > threshold).float()
                true = all_labels[:, label_idx]
                
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
        
        return torch.tensor(optimal_thresholds).to(self.device)

    def evaluate(self, loader, phase, epoch=None, thresholds=None):
        """Evaluate model on given loader"""
        if thresholds is None:
            thresholds = torch.ones(self.model.num_classes).to(self.device) * 0.5
        
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.model.eval()
        
        total_samples = 0
        total_loss = 0.0
        exact_matches = 0
        total_true_positives = 0
        total_false_positives = 0
        total_false_negatives = 0
        
        with torch.no_grad():
            for img, labels in loader:
                img = img.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(img)
                batch_loss = criterion(outputs, labels)
                total_loss += batch_loss.item()
                
                predictions = (torch.sigmoid(outputs) > thresholds[None, :]).float()
                
                exact_matches += (predictions == labels).all(dim=1).sum().item()
                
                true_positives = (predictions * labels).sum(dim=1)
                false_positives = (predictions * (1 - labels)).sum(dim=1)
                false_negatives = ((1 - predictions) * labels).sum(dim=1)
                
                total_true_positives += true_positives.sum().item()
                total_false_positives += false_positives.sum().item()
                total_false_negatives += false_negatives.sum().item()
                
                total_samples += img.size(0)
        
        metrics = {
            'loss': total_loss / total_samples,
            'exact_match_ratio': exact_matches / total_samples,
            'precision': total_true_positives / (total_true_positives + total_false_positives + 1e-10),
            'recall': total_true_positives / (total_true_positives + total_false_negatives + 1e-10)
        }
        
        metrics['micro_f1'] = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'] + 1e-10)
        
        if epoch is not None:
            for metric_name, metric_value in metrics.items():
                self.writer.add_scalar(f"{phase}/{metric_name}", metric_value, epoch)

            self.writer.flush()
        
        return metrics

    def save_checkpoint(self, epoch, thresholds):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_f1': self.best_val_f1,
            'epoch': epoch,
            'thresholds': thresholds,
            'model_parameters': self.model.get_parameters()
        }
        torch.save(checkpoint, f'{self.model.__class__.__name__}_checkpoint_f1_{self.best_val_f1:.3f}.pt')

    def train(self):
        """Main training loop"""
        optimal_thresholds = torch.ones(self.model.num_classes).to(self.device) * 0.5

        print(f"Starting training from epoch {self.start_epoch}/{self.num_epochs}")
        remaining_epochs = self.num_epochs - self.start_epoch
        print(f"Remaining epochs: {remaining_epochs}")
        
        for epoch in range(self.start_epoch, self.num_epochs): 
            print(f"Starting epoch: {epoch}")
            
            # Training phase
            self.model.train()
            losses = []
            
            for img, label in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                img = img.to(self.device)
                label = label.to(self.device)
                
                self.optimizer.zero_grad()
                
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = self.model(img)
                    label = label.float()
                    loss = self.criterion(outputs, label)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                losses.append(loss.item())
            
            epoch_loss = sum(losses) / len(losses)
            self.writer.add_scalar("train/loss", epoch_loss, epoch)
            
            # Validation phase
            optimal_thresholds = self.find_optimal_thresholds()
            metrics = self.evaluate(self.val_loader, "validation", epoch, optimal_thresholds)
            
            # Early stopping check
            self.early_stopping(metrics['loss'], self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            self.scheduler.step()
            
            # Save best model
            if metrics['micro_f1'] > self.best_val_f1:
                self.best_val_f1 = metrics['micro_f1']
                self.save_checkpoint(epoch, optimal_thresholds)

         # Final evaluation on test set
        test_metrics = self.evaluate(self.test_loader, "test", None, self.optimal_thresholds)
        print("Final test metrics:", test_metrics)
        return test_metrics        

def load_checkpoint(model, checkpoint_path, device):
    """Load model from checkpoint"""
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint