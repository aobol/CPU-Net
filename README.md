# CPU-Net

Cyclic Positional U-Net (CPU-Net) is a transfer‑learning framework that learns an ad‑hoc translation between simulated and measured HPGe detector pulses.

## Key Highlights

- Cycle GAN backbone couples a REN (sim → data) and an IREN (data → sim), enforcing cycle‑ and identity‑consistency.
- Positional U‑Net generators with layer‑wise positional encodings accurately translate pulse shapes.
- Attention‑augmented RNN discriminators evaluate translations and guide adversarial training.
- Translated pulses reproduce ensemble distributions – current amplitude, drift time, tail slope – much better than raw simulations.
- Architecture can be adapted to other scientific domains where noise convolution/de‑convolution is required.

---

## Usage

### Files and Directories

- `network.py`   CPU‑Net model definitions (Positional UNet + RNN disc).
- `dataset.py`    `SplinterDataset` for loading & preprocessing pickled pulses.
- `tools.py`      DSP helpers, IoU / ROC metrics, validation utilities.
- `TrainAndPlot.ipynb`  Training notebook
- `Analysis.ipynb`      Validation notebook for unseen data.

---

## Datasets & Model Weights ( Zenodo )

Data is packaged in Zenodo DOI [10.5281/zenodo.15311838](https://zenodo.org/records/15311838).

Inventory

- `fep_REN.pt`, `fep_IREN.pt`  – generators (sim → data, data → sim)
- `fep_netD_A.pth`, `fep_netD_B.pth`  – discriminators
- `_wf_ornl.pickle`  – real FEP / SEP / DEP detector pulses
- `_wf_sim.pickle`   – matching Geant4 + Siggen simulations

Each pickle entry:
```python
{
    "wf":     np.ndarray,  # (800,) aligned & normalised waveform
    "tp0":    int,         # index of 0 % rise
    "energy": float        # calibrated energy (keV)
}
```

#### Data
 **Detector pulses** were recorded with an inverted‑coax HPGe detector (serial V06643A) during a –228Th–flood calibration at Oak Ridge National Laboratory. Signals were digitised by FlashCam module; pygama handled conversion to HDF5.
 
 **Simulated pulses** originate from a Geant4 model of the same setup. Energy deposits were fed to siggen to generate raw charge‑collection pulses.

---

## Training

Open `TrainAndPlot.ipynb` and follow the cells for:
1. Data preprocessing ↔ `SplinterDataset` cuts
2. Model initialisation (`PositionalUNet`, `RNN`)
3. Training loop (Cycle‑GAN losses, schedulers)
4. Check‑point saving to `model_weights/`

 GPU NVIDIA A100 (40 GB) – one hour for 7000 iterations.

---

## Analysis

Notebook `Analysis.ipynb` evaluates a trained REN/IREN on held‑out data:

- Pulse translation and overlay plots.
- Histograms of real vs sim vs translated pulses.
- IoU scores for current amplitude, drift time, tail slope.

---

## Training & Analysis Strategy

| Stage | Dataset | Rationale |
|-------|---------|-----------|
| Training | FEP (2614 keV full‑energy peak) – mix of single‑ & multi‑site events | Abundant stats + diverse topologies lets REN learn full electronics response. |
| Validation – single‑site | DEP (double‑escape peak) | Majority single‑site pulses test preservation of single site events. |
| Validation – multi‑site | SEP (single‑escape peak) | Majority multi‑site pulses stress reproduction of multi site events. |

### Validation metrics

- Maximum current amplitude (I<sub>max</sub>) – distinguishes single vs multi‑site; REN output should match detector distribution.
- Drift time (T<sub>drift</sub>) – time between 1 % and 100 % rise; verifies realistic charge‑collection times after translation.

---

## License

CPU‑Net is released under the MIT License – see `LICENSE`.

---

## Contact and support

| Name | Role | Email |
|------|------|-------|
| Kevin Bhimani | Lead developer | kevinhbhimani@gmail.com |
| Aobo Li | Project initiator & mentor | aol002@ucsd.edu |

