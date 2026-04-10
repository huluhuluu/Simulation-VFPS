"""
Model Evaluation Module
"""

import torch
from typing import List, Tuple, Dict

from src.splitnn import SplitNN


def evaluate(splitnn: SplitNN, test_set: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]], 
             device: torch.device) -> float:
    """Evaluate model accuracy
    
    Args:
        splitnn: Split neural network
        test_set: Test set
        device: Computation device
        
    Returns:
        Accuracy
    """
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data_ptr, label in test_set:
            label = label.to(device)
            # Ignore communication time and outputs
            pred, _, _ = splitnn.predict(data_ptr)
            
            pred_labels = pred.argmax(dim=1).cpu().numpy()
            true_labels = label.cpu().numpy()
            
            correct += (pred_labels == true_labels).sum()
            total += len(true_labels)
    
    return correct / total if total > 0 else 0.0
