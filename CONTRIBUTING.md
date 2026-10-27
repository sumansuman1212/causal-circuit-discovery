# Contributing to Causal Circuit Discovery

Thank you for your interest in contributing to **Automated Discovery and Causal Validation of Minimal Transformer Circuits**.

This repository is designed for mechanistic interpretability and AI safety research. We welcome contributions that improve mathematical rigor, code quality, testing coverage, algorithm efficiency, and experimental reproducibility.

---

## Code of Conduct & Research Integrity

All contributors are expected to uphold the highest standards of scientific and engineering integrity:
- **Zero Fabrication:** Never commit synthetic, fabricated, or assumed experimental metrics into reports, docstrings, or papers. All quantitative claims must be directly traceable to saved JSON/CSV outputs and reproducible seeds.
- **Explicit Invariants:** Maintain clear assertions and type hints. Do not hide failing interventions or edge cases.
- **Hardware-Aware Execution:** Default to CPU-safe evaluation in tests and CLI defaults so that reproduction does not require local GPU hardware.

---

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sumansuman1212/causal-circuit-discovery.git
   cd causal-circuit-discovery
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux / macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the test suite:**
   ```bash
   pytest tests/
   ```

---

## Submitting Pull Requests

1. **Create a topic branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Ensure all existing tests pass:**
   ```bash
   pytest -v
   ```
3. **Add unit or integration tests** for any new functionality in `tests/`.
4. **Follow style guidelines:** Use clean, readable Python with standard PEP 8 formatting and meaningful docstrings.
5. **Open a Pull Request** describing:
   - What the change achieves.
   - The rationale for non-obvious design choices.
   - Verification evidence (command outputs or test results).
