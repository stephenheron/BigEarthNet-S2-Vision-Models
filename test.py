import torch
import time
import argparse
from dataset import BigEarthNetDataSet
from vit import ViT
from cnn import CNN  # Added CNN import
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from convnext import ConvNeXt  

torch.set_num_threads(mp.cpu_count())
torch.set_float32_matmul_precision('high')

def evaluate_model(model, test_loader, thresholds, device):
    model.eval()
    total_samples = 0
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0
    exact_matches = 0
    
    total_time = 0
    total_batches = 0
    
    total_steps = len(test_loader)
    print_interval = max(total_steps // 20, 1)
    
    print("\nStarting evaluation...")
    print("Progress | Precision | Recall | F1 Score | Exact Match | Images/sec | MS/image")
    print("-" * 75)
    
    with torch.no_grad():
        for batch_idx, (img, labels) in enumerate(test_loader):
            img = img.to(device)
            labels = labels.to(device)
            
            start_time = time.perf_counter()
            outputs = model(img)
            end_time = time.perf_counter()
            
            batch_time = end_time - start_time
            total_time += batch_time
            total_batches += 1
            
            predictions = (torch.sigmoid(outputs) > thresholds[None, :]).float()
            exact_matches += (predictions == labels).all(dim=1).sum().item()
            
            true_positives = (predictions * labels).sum(dim=1)
            false_positives = (predictions * (1 - labels)).sum(dim=1)
            false_negatives = ((1 - predictions) * labels).sum(dim=1)
            
            total_true_positives += true_positives.sum().item()
            total_false_positives += false_positives.sum().item()
            total_false_negatives += false_negatives.sum().item()
            
            total_samples += img.size(0)
            
            if (batch_idx + 1) % print_interval == 0 or (batch_idx + 1) == total_steps:
                precision = total_true_positives / (total_true_positives + total_false_positives + 1e-10)
                recall = total_true_positives / (total_true_positives + total_false_negatives + 1e-10)
                f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
                exact_match_ratio = exact_matches / total_samples
                images_per_second = total_samples / total_time
                ms_per_image = (total_time / total_samples) * 1000
                
                progress = (batch_idx + 1) / total_steps * 100
                print(f"{progress:7.1f}% | {precision:.4f} | {recall:.4f} | {f1_score:.4f} | "
                      f"{exact_match_ratio:.4f} | {images_per_second:8.1f} | {ms_per_image:8.1f}")
    
    final_metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'exact_match_ratio': exact_match_ratio,
        'images_per_second': images_per_second,
        'ms_per_image': ms_per_image,
        'avg_batch_time': total_time / total_batches
    }
    
    return final_metrics

def main():
    parser = argparse.ArgumentParser(description='Test ViT or CNN model on BigEarthNet dataset')
    parser.add_argument('--model', type=str, choices=['vit', 'cnn', 'convnext'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model parameters from checkpoint
    model_params = checkpoint['model_parameters']
    
    # Create model based on saved parameters
    if args.model == 'vit':
        model = ViT(
            img_width=model_params.get('img_width', 120),
            img_channels=model_params.get('img_channels', 5),
            patch_size=model_params.get('patch_size', 8),
            d_model=model_params.get('embedding_dim', 256),
            num_heads=model_params.get('num_heads', 8),
            num_layers=model_params.get('num_layers', 6),
            num_classes=model_params.get('num_classes', 19),
            ff_dim=model_params.get('ff_dim', 1536),
        ).to(device)
    elif args.model == 'cnn' :  # CNN
        model = CNN(
            img_channels=model_params.get('img_channels', 5),
            num_classes=model_params.get('num_classes', 19)
        ).to(device)
    elif args.model == 'convnext' :  # CNN
        model = ConvNeXt(
            img_channels=model_params.get('img_channels', 5),
            num_classes=model_params.get('num_classes', 19)
        ).to(device)
    else:
        raise SystemExit("No model available.")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    # Get thresholds with a default if not found
    # Get thresholds with a default if not found
    thresholds = checkpoint.get('thresholds', torch.tensor([0.5] * model_params.get('num_classes', 19)).to(device))
    
    # Print model parameters
    print("\nModel Parameters:")
    params = model.get_parameters()
    print(f"Model Type: {params['model_type']}")
    print(f"Total Parameters: {params['total_parameters']:,}")
    
    test_dataset = BigEarthNetDataSet('test')
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print("Starting evaluation...")
    metrics = evaluate_model(model, test_loader, thresholds, device)
    
    print("\nTest Set Metrics:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print(f"Exact Match Ratio: {metrics['exact_match_ratio']:.4f}")
    
    print("\nSpeed Metrics:")
    print(f"Images per second: {metrics['images_per_second']:.2f}")
    print(f"Milliseconds per image: {metrics['ms_per_image']:.2f}")
    print(f"Average batch time: {metrics['avg_batch_time']*1000:.2f} ms")

if __name__ == "__main__":
    main()