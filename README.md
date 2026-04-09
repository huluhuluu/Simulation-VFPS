# Simulation-VFPS

A simulation code for paper **VF-PS: How to Select Important Participants in Vertical Federated Learning, Efficiently and Securely?**

This project implements dynamic participant selection for Vertical Federated Learning (VFL) using mutual information estimation with group testing.

## Reference

- **Paper**: [VF-PS: How to Select Important Participants in Vertical Federated Learning, Efficiently and Securely?](https://arxiv.org/abs/2205.12731)
- **Original Repository**: [Dynamic-VFPS](https://github.com/r-gheda/Dynamic-VFPS)

## Versions

| Version | File | Description |
|---------|------|-------------|
| **Legacy** | `test.py` | Old version implementation modified from the reference repository |
| **New** | `test_gpu.py` | Updated version with CUDA/GPU support, compatible with newer PyTorch versions |

## Installation

```bash
# Create conda environment
conda env create -f environment_cuda.yml
conda activate vfps-gpu

# Or use pip
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Default parameters (dynamic mode, plaintext)
python test_gpu.py

# Custom parameters
python test_gpu.py --epochs 50 --clients 10 --selected 6

# With encryption
python test_gpu.py --encryption tenseal
python test_gpu.py --encryption paillier

# Static MI mode (select clients once before training)
python test_gpu.py --mi-mode static --mi-ratio 0.111
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--lr` | 0.001 | Learning rate |
| `--batch-size` | 256 | Batch size |
| `--local-epochs` | 1 | Local iterations per batch |
| `--clients` | 10 | Number of total clients |
| `--selected` | 6 | Number of selected clients |
| `--n-tests` | 5 | Number of group tests |
| `--mi-mode` | dynamic | MI mode: `dynamic` or `static` |
| `--mi-ratio` | 0.111 | Data ratio for MI estimation (static mode) |
| `--encryption` | plaintext | Encryption: `plaintext`, `paillier`, `tenseal` |
| `--bandwidth` | 300 | Bandwidth in Mbps |

## Time Statistics

The implementation provides detailed time breakdown:

| Component | Description |
|-----------|-------------|
| **Train** | Forward + backward computation time |
| **Comm** | Communication time (plaintext activation + gradient) |
| **MI Compute** | MI estimation computation time |
| **MI Comm** | Encrypted data transmission time for MI estimation |

### Communication Flow

1. **Client Selection Phase**:
   - Clients send **encrypted** raw data to server
   - Server computes MI and selects participants

2. **Model Training Phase**:
   - Forward: Clients send **plaintext** activations to server
   - Backward: Server sends **plaintext** gradients to clients

## Project Structure

```
Dynamic-VFPS/
├── test.py              # Legacy version
├── test_gpu.py          # New version (GPU support)
├── src/
│   ├── models/          # Neural network models
│   ├── transmission/    # Encryption implementations
│   │   ├── plaintext.py
│   │   ├── paillier/
│   │   └── tenseal/
│   └── utils/           # Utility functions
├── datasets/            # Dataset storage
└── README.md
```

## Model Architecture

- **Client Model**: ResNet18 (adapted for 28x28 single-channel images)
- **Server Model**: Fully connected layers for classification
- **Dataset**: Fashion-MNIST (vertical partition)

### Data Partition

Images are vertically partitioned by columns:
- Each client receives `28 × (28/n_clients)` pixels
- For 10 clients: each gets `28 × 2 = 56` features per sample


## License

MIT License
