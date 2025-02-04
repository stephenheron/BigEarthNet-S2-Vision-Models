import argparse
import torch
from dataset import BigEarthNetDataSet
from training_framework import ModelTrainer, load_checkpoint
from vit import ViT  

def main():
    parser = argparse.ArgumentParser(description='Train ViT or CNN models on BigEarthNet')
    parser.add_argument('--model', type=str, choices=['vit', 'cnn'], required=True,
                      help='Model architecture to train (vit or cnn)')
    parser.add_argument('--checkpoint', help='Path to checkpoint file to resume training from', default=None)
    
    args = parser.parse_args()

    batch_size = 256

    # Set device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load datasets
    train_dataset = BigEarthNetDataSet('train')
    test_dataset = BigEarthNetDataSet('test')
    validation_dataset = BigEarthNetDataSet('validation')

    # Create data loaders
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
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Model parameters
    img_width = 120
    img_channels = 5
    num_classes = 19

    num_epochs = None
    learning_rate = None
    weight_decay = None


    # Create model based on architecture choice
    if args.model == 'vit':
        num_epochs = 40
        learning_rate = 4e-3
        weight_decay = 2e-4

        model = ViT(
            img_width=img_width,
            img_channels=img_channels,
            patch_size=8,
            d_model=256,
            num_heads=8,
            num_layers=6,
            num_classes=num_classes,
            ff_dim=1536
        ).to(device)
    else:  # CNN
        pass
        #model = CNNModel(
        #    img_channels=img_channels,
        #    num_classes=num_classes
        #).to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        model, checkpoint = load_checkpoint(model, args.checkpoint, device)
        print(f"Loaded checkpoint from {args.checkpoint}")
        print(f"Previous best F1: {checkpoint['best_val_f1']:.4f}")

    # Print model parameters
    print("\nModel Parameters:")
    params = model.get_parameters()
    print(f"Model Type: {params['model_type']}")
    print(f"Total Parameters: {params['total_parameters']:,}")

    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs
    )

    # Train model
    print("\nStarting training...")
    metrics = trainer.train()

    # Print final metrics
    print("\nFinal Test Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")

if __name__ == "__main__":
    main()