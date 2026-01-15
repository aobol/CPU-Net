# CPU-Net

**CPU-Net (Cyclic Positional U-Net)** is a transfer-learning framework that learns an ad-hoc translation between **simulated** and **measured** HPGe detector pulses. It is designed to emulate readout electronics effects in pulse-shape simulations using an unpaired, CycleGAN-style training strategy.

**Paper:** *CycleGAN-driven transfer learning for electronics response emulation in high-purity germanium detectors* (Machine Learning: Science and Technology, 2026).  
DOI: [10.1088/2632-2153/ae3052](https://doi.org/10.1088/2632-2153/ae3052)

**Data & pretrained weights (Zenodo):**  
DOI: [10.5281/zenodo.15311838](https://doi.org/10.5281/zenodo.15311838)

---

## Key highlights

- **CycleGAN backbone** couples a **REN** (sim → data) and an **IREN** (data → sim), enforcing cycle- and identity-consistency.
- **Positional U-Net generators** use layer-wise positional encodings to better preserve pulse timing/structure (esp. tails).
- **Attention-augmented RNN discriminators** evaluate translations and guide adversarial training.
- Translated pulses reproduce ensemble distributions (e.g., **current amplitude**, **drift time**, **tail slope**) better than raw simulations.
- Architecture is adaptable to other scientific time-series domains where noise convolution / deconvolution is required.

---

## Quickstart

### 1) Create the environment (recommended)

```bash
conda env create -f environment.yml
conda activate cpu-net
```

### 2) Install CPU-Net (editable)

```bash
pip install -e . --no-deps
```

### 3) Run tests

```bash
pip install pytest
pytest -q
```

> Note: We pin **NumPy < 2** (see `environment.yml` / `pyproject.toml`) to avoid incompatibilities with compiled dependencies in common scientific Python stacks.

---

## Data & model weights (Zenodo)

Data and pretrained weights are packaged on Zenodo: [10.5281/zenodo.15311838](https://zenodo.org/records/15311838).

### Contents

- `fep_REN.pt`, `fep_IREN.pt` — generators (sim → data, data → sim)
- `fep_netD_A.pth`, `fep_netD_B.pth` — discriminators
- `_wf_ornl.pickle` — real FEP / SEP / DEP detector pulses
- `_wf_sim.pickle` — matching Geant4 + siggen simulations

Each pickle entry:

```python
{
    "wf":     np.ndarray,  # (800,) aligned & normalized waveform
    "tp0":    int,         # index of 0% rise
    "energy": float        # calibrated energy (keV)
}
```

#### Data
 **Detector pulses** were recorded with an inverted‑coax HPGe detector (serial V06643A) during a –228Th–flood calibration at Oak Ridge National Laboratory. Signals were digitised by FlashCam module; pygama handled conversion to HDF5.
 
 **Simulated pulses** originate from a Geant4 model of the same setup. Energy deposits were fed to siggen to generate raw charge‑collection pulses.

---

## Running training and analysis

The easiest way to reproduce figures/metrics is via the notebooks:

- `notebooks/Training.ipynb` — training loop (CycleGAN training of REN/IREN)
- `notebooks/Validation.ipynb` — validation on held-out datasets and distribution-level metrics


---

## Repository layout

**Core package code:**
- `src/cpunet/network.py` — CPU-Net model definitions (Positional U-Net + discriminator(s))
- `src/cpunet/dataset.py` — dataset loading & preprocessing (pickled pulses)
- `src/cpunet/tools.py` — DSP helpers, metrics (IoU / ROC), plotting utilities

**Notebooks:**
- `notebooks/Training.ipynb`
- `notebooks/Validation.ipynb`

**Tests:**
- `tests/test_imports.py` — import/smoke test for packaging integrity

---

## Training & validation strategy (as used in the paper)

| Stage | Dataset | Rationale |
|------|---------|-----------|
| Training | FEP (2614 keV full-energy peak) — mix of single- & multi-site events | Abundant statistics and diverse topologies; translation learned as a global electronics-like effect. |
| Validation (single-site dominated) | DEP (double-escape peak) | Tests preservation of single-site pulse features. |
| Validation (multi-site dominated) | SEP (single-escape peak) | Stress-tests reproduction of multi-site behavior. |

Typical training runtime: **~1 GPU-hour** on an **NVIDIA A100 (40 GB)** for **7000 iterations**.

### Validation metrics (examples)
- **Maximum current amplitude (Imax):** distinguishes single vs multi-site; translated pulses should match detector distribution.
- **Drift time (Tdrift):** time between 1% and 100% rise; checks realistic charge-collection timing after translation.
- (Optional) Tail decay constant **τ**: distribution-level check of RC decay behavior.

---

## Citation

If you use this code or the associated data in academic work, please cite the paper:

- **Kevin Bhimani**, Julieta Gruszko, Morgan Clark, John Wilkerson, Aobo Li,  
  *CycleGAN-driven transfer learning for electronics response emulation in high-purity germanium detectors*,  
  *Machine Learning: Science and Technology* (2026). DOI: **10.1088/2632-2153/ae3052**

Zenodo dataset DOI: **10.5281/zenodo.15311838**

(Also see `CITATION.cff`.)

---

## License

Released under the **MIT License** (see `LICENSE`).

---

## Contact

| Name | Role / Contribution | Email |
|------|----------------------|-------|
| Kevin Bhimani | Lead developer | kevinhbhimani@gmail.com |
| Aobo Li | Project initiator & mentor | aol002@ucsd.edu |
| Julieta Gruszko | Principal investigator | jgruszko@unc.edu |
