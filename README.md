## ⚙️ Project Architecture
This repository follows a production-ready PyTorch Lightning structure for modularity and reproducibility.

- **Config:** Managed via Hydra/YAML for easy hyperparameter sweeps.
- **DataModules:** Handles sensor preprocessing and train/val/test splits.
- **LightningModules:** Decouples the physics-informed training logic from the hardware.

## 🚀 Getting Started
### Training
```bash
python src/pipeline/train.py --config config/baseline.yaml