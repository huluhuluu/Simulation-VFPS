"""
Utility Functions Module
"""

import argparse
import math
import torch


def digamma(x: float) -> float:
    """Digamma function (logarithmic derivative of Gamma function)
    
    Args:
        x: Input value
        
    Returns:
        Digamma function value
    """
    if x == 0:
        return float('-inf')
    return math.log(x) - 0.5 / x


def get_device() -> torch.device:
    """Get computation device
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def parse_args() -> argparse.Namespace:
    """Parse command line arguments
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description='Dynamic-VFPS GPU Test')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, default='fashion-mnist',
                       choices=['fashion-mnist', 'cifar-10'],
                       help='Dataset to use: fashion-mnist or cifar-10')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--local-epochs', type=int, default=1, help='Local iterations per batch')
    
    # Client parameters
    parser.add_argument('--clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--selected', type=int, default=6, help='Number of selected clients')
    
    # MI estimation parameters
    parser.add_argument('--n-tests', type=int, default=5, help='Number of group tests')
    parser.add_argument('--k-nn', type=int, default=3, help='KNN k value')
    parser.add_argument('--mi-mode', type=str, default='static',
                       choices=['dynamic', 'static'],
                       help='MI mode: dynamic (every step) or static (once at start)')
    parser.add_argument('--mi-ratio', type=float, default=1/9,
                       help='Ratio of training data for MI estimation (static mode)')
    
    # Communication parameters
    parser.add_argument('--bandwidth', type=int, default=300, help='Bandwidth in Mbps')
    
    # Encryption parameters
    parser.add_argument('--encryption', type=str, default='plaintext',
                       choices=['plaintext', 'paillier', 'tenseal'],
                       help='Encryption method')
    
    return parser.parse_args()
