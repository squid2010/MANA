# MANA 🔬

**Molecular Architecture for Near-infrared Absorbers**

A deep learning framework for predicting photophysical properties of photosensitizers using equivariant graph neural networks. MANA predicts absorption wavelengths (λ_max), fluorescence properties, and singlet oxygen quantum yields (Φ_Δ) in solvent-aware environments.

## 🎯 Overview

MANA is a multi-task neural network architecture built on the PaiNN (Polarizable Atom Interaction Neural Network) backbone. It processes molecular structures and solvent information to predict key photophysical properties relevant for applications like photodynamic therapy (PDT) and near-infrared imaging.

**Key Features:**
- **E(3)-equivariant** architecture respecting physical symmetries
- **Solvent-aware** predictions accounting for solvatochromic effects
- **Multi-phase training** strategy with progressive fine-tuning
- **3D conformer generation** from SMILES strings
- Predicts λ_max (absorption), fluorescence wavelengths, and Φ_Δ quantum yields

## 🏗️ Architecture

MANA uses a dual-stream architecture:

```
Solute Molecule → Embedding → PaiNN Layers → Global Pooling ↘
                                                                → Interaction → Task Heads
Solvent Shell   → Embedding → PaiNN Layers → Global Pooling ↗                   ├─ λ_max
                                                                                └─ Φ_Δ
```

**Core Components:**
- **Embedding Layer**: Converts atom types to learnable vectors (118 atom types)
- **Radial Basis Functions (RBF)**: Encodes distance information using Gaussian RBFs
- **PaiNN Layers**: 4 stacked equivariant message-passing layers (128-dim hidden)
- **Task Heads**: Specialized prediction heads for different photophysical properties

## 📊 Training Strategy

MANA uses a **3-phase training approach**:

1. **Phase 1 - Absorption (λ_max)**: Train backbone and λ_max head on absorption data
2. **Phase 2 - Fluorescence**: Fine-tune with fluorescence emission data
3. **Phase 3 - Quantum Yield (Φ_Δ)**: Fine-tune phi head with optional backbone freezing

Each phase supports:
- Molecular ID-based splitting (prevents conformer leakage)
- Early stopping with validation monitoring
- Learning rate scheduling (ReduceLROnPlateau)
- Loss component tracking

## 🚀 Installation

### Requirements
```bash
# Core dependencies
torch>=2.0.0
torch-geometric>=2.3.0
rdkit>=2023.3.1
h5py>=3.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scipy>=1.10.0
tqdm>=4.65.0

# Optional for visualization
graphviz>=0.20.0
```

### Setup
```bash
git clone https://github.com/squid2010/MANA.git
cd MANA
pip install -r requirements.txt  # Create this from dependencies above
```

## 📖 Usage

### Training

```python
from scripts.train import train_phase
from pathlib import Path

# Phase 1: Absorption wavelength
hyperparams_p1 = {
    "tasks": ["lambda"],
    "max_epochs": 200,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "early_stopping_patience": 30,
}

train_phase(
    phase_name="Phase1_Absorption",
    hyperparams=hyperparams_p1,
    dataset_path="data/lambda/lambda_only_data.h5",
    save_dir="models/phase1",
)

# Phase 2: Fluorescence (load Phase 1 weights)
hyperparams_p2 = {
    "tasks": ["lambda", "fluorescence"],
    "max_epochs": 150,
    "learning_rate": 5e-5,
    "weight_decay": 5e-4,
    "early_stopping_patience": 30,
}

train_phase(
    phase_name="Phase2_Fluorescence", 
    hyperparams=hyperparams_p2,
    dataset_path="data/phi/phidelta_data.h5",
    save_dir="models/phase2",
    load_path="models/phase1/best_model.pth",
)
```

### Evaluation

```python
from scripts.evaluation.evaluate import run_inference, generate_report

# Run inference on test set
predictions = run_inference(
    model_path="models/phase2/best_model.pth",
    dataset_path="data/test_data.h5",
    tasks=["lambda", "phi"],
)

# Generate evaluation report with plots
generate_report(predictions, output_dir="results/evaluation")
```

### Molecular Screening

```python
from scripts.screening.screen import screen_molecules

# Screen candidate molecules from SMILES
candidates = [
    "c1ccc2c(c1)c1ccccc1n2CCO",  # Example BODIPY-like structure
    "c1ccc2c(c1)sc1ccccc12",      # Thioxanthene derivative
]

results = screen_molecules(
    smiles_list=candidates,
    solvents=["water", "ethanol", "chloroform"],
    model_path="models/phase2/best_model.pth",
    output_path="results/screening_results.h5",
)
```

## 📁 Project Structure

```
MANA/
├── data/                      # Dataset storage (HDF5 files)
├── img/                       # Figures and visualizations
├── models/                    # Trained model checkpoints
├── notebooks/                 # Jupyter/Colab training notebooks
├── results/                   # Evaluation outputs and plots
├── scripts/
│   ├── data/                  # Dataset construction
│   │   ├── build_phi_dataset.py
│   │   ├── build_pretrain_dataset.py
│   │   └── dataset.py         # DatasetConstructor class
│   ├── evaluation/            # Model evaluation
│   │   ├── evaluate.py        # Main evaluation script
│   │   └── trained_vs_untrained.py
│   ├── miscellaneous/         # Utilities
│   │   ├── make_model_image.py
│   │   └── visualize_mol.py
│   ├── model/                 # Neural network architecture
│   │   ├── mana_model.py      # MANA model definition
│   │   └── training_engine.py # Training loop
│   ├── screening/             # Virtual screening tools
│   │   ├── screen.py
│   │   └── analyze_results.py
│   └── train.py               # Training pipeline
└── notes/                     # Development notes
```

## 📈 Performance Metrics

MANA evaluates performance using:
- **λ_max**: RMSE, MAE, R² correlation
- **Φ_Δ**: Accuracy (binned), Kendall's τ ranking correlation
- **Solvatochromic shifts**: Prediction of λ_vacuum vs λ_solvated
- **Combined utility**: Gaussian(λ) × clipped(Φ_Δ) for NIR scoring

## 🔬 Datasets

MANA supports HDF5 datasets with the following structure:

```python
# Required fields
- smiles: bytes         # SMILES strings
- positions: float32    # 3D coordinates (N, max_atoms, 3)
- atomic_numbers: int32 # Atomic numbers (N, max_atoms)
- lambda_abs: float32   # Absorption wavelength (nm)
- lambda_flu: float32   # Fluorescence wavelength (nm, optional)
- phi_delta: float32    # Quantum yield (0-1, optional)
- solvent_smiles: bytes # Solvent SMILES (optional)
- mol_id: int32         # Molecule identifier for splitting
```

### Building Datasets

```bash
# Build absorption dataset
python scripts/data/build_pretrain_dataset.py \
    --input data/raw/absorption_data.csv \
    --output data/lambda/lambda_only_data.h5

# Build quantum yield dataset  
python scripts/data/build_phi_dataset.py \
    --input data/raw/wilkinson_dataset.csv \
    --output data/phi/phidelta_data.h5 \
    --num-conformers 10
```

## 🎨 Visualization

Generate architecture diagrams:
```bash
python scripts/miscellaneous/make_model_image.py \
    --out img/mana_architecture.png \
    --format png \
    --layers 4 \
    --nodes 6
```

Visualize molecular predictions:
```bash
python scripts/miscellaneous/visualize_mol.py \
    --smiles "c1ccc2c(c1)sc1ccccc12" \
    --model models/phase2/best_model.pth
```

## 🤝 Contributing

This project was developed as part of a research initiative. Contributions, issues, and feature requests are welcome!

## 📄 License

[Specify your license here]

## 🙏 Acknowledgments

- Built using PyTorch Geometric and RDKit
- PaiNN architecture inspired by Schütt et al. (2021)
- Dataset processing leverages the Deep4Chem and Wilkinson photosensitizer databases

## 📧 Contact

[Your contact information]

## 🔗 References

- **PaiNN**: Schütt et al., "Equivariant message passing for the prediction of tensorial properties and molecular spectra", ICML 2021
- **PyTorch Geometric**: Fey & Lenssen, "Fast Graph Representation Learning with PyTorch Geometric", 2019

---

**Note**: This project is under active development. Some features may be experimental.
