# Automated Discovery and Causal Validation of Minimal Transformer Circuits

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An interpretability research framework to automatically identify, ablate, prune, and causally validate minimal Transformer circuits, with cross-seed stability and out-of-distribution (OOD) evaluations.

---

## 1. Quickstart & Installation

```bash
# Clone the repository
git clone [https://github.com/sumansuman1212/causal-circuit-discovery.git](https://github.com/sumansuman1212/causal-circuit-discovery.git)
cd causal-circuit-discovery

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run test suite to verify invariants
pytest tests/
