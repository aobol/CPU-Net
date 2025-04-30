# CPU-Net

Cyclic Positional U-Net (CPU-Net) is a transfer learning model for ad-hoc pulse shape translation in HPGe detectors.

## Key Highlights
- CPU-Net utilizes a cycle GAN architecture coupled with positional encoding to accurately translate pulse shapes. 
- Utilizes U-Net architecture with positional encoding for its generators (A2B and B2A). This allows for accurate pulse translation between real and simulated data while maintaining cycle and identity consistency.
- Discriminators (DA and DB) are Recurrent Neural Networks (RNN) with attention mechanisms, evaluating translated pulses and optimizing the performance of generators through adversarial training.
- CPU-Net accurately translates the simulated pulses to match data pulses, while reproducing the ensemble distribution of the data.
- Although designed for HPGe detectors, CPU-Net's architecture is adaptable to different scientific domains for convoluting and deconvoluting noise.


## Usage

### Files and Directories

- `network.py`: Contains the CPU-Net model architecture.
- `dataset.py`: Defines a function for loading and preprocessing pulse data into Pytorch Dataloader.
- `tools.py`: Includes utilities for data processing, pulse analysis, and evaluation metrics.
- `TrainAndPlot.ipynb`: Jupyter notebook for training the model and visualizing results.
- `Analysis.ipynb`: Notebook for model performance analysis on unseen data.

## Datasets & Model Weights ( Zenodo )

All training/validation pulses **and** pre-trained CPU-Net weights are archived in a single Zenodo record:

> **Zenodo DOI** **10.5281/zenodo.15311838**  
> <https://zenodo.org/records/15311838>

### Contents   (≈ 1.9 GB total)
| File | Purpose | Size |
|------|---------|------|
| **Model weights** |  |
| `fep_REN.pt` / `fep_IREN.pt` | Generators: simulation → data (**REN**) and data → simulation (**IREN**) | 2 × 78 MB |
| `fep_netD_A.pth` / `fep_netD_B.pth` | Discriminators for data and simulation domains | 2 × 0.5 MB |
| **Detector pulses** |  |
| `fep_wf_ornl.pickle` | 107 k full-energy-peak (FEP) pulses | 934 MB |
| `sep_wf_ornl.pickle` | 3 k single-escape (SEP) pulses | 151 MB |
| `dep_wf_ornl.pickle` | 1.2 k double-escape (DEP) pulses | 83 MB |
| **Siggen simulations** |  |
| `fep_wf_sim.pickle` | 110 k FEP simulations | 461 MB |
| `sep_wf_sim.pickle` | SEP simulations | 53 MB |
| `dep_wf_sim.pickle` | DEP simulations | 13 MB |

Each pickle entry is:

```python
{
    "wf":     np.ndarray,  # (800,) aligned, normalised waveform
    "tp0":    int,         # index of 0 % rise
    "energy": float        # calibrated energy in keV
}

```

### Training

Open `TrainAndPlot.ipynb` and follow the steps for data loading, model training, and visualization of results. The notebook outlines the training process, including:

- Data preprocessing.
- Model initialization.
- Training loop execution.
- Model saving.

- **GPU**: Nvidia A100 GPUs
- **RAM usage**: About 6Gb
- **Training Time**: 60 mins
### Analysis

Use `Analysis.ipynb` to evaluate the model on test data. This notebook allows for:

- Pulse transformation through the CPU-Net.
- Comparison of real, simulated, and transformed pulses.
- Visualization and statistical analysis of the results.
- Comparison of key pulse parameters such as current amplitude and tail slopw.

## License

This project is released under the MIT License - see the LICENSE file for details.

## Contact and Support

For questions, feedback, or contributions to the CPU-Net project, please feel free to reach out. You can contact us via email:

- **Kevin Bhimani**
  - Email: [kevinhbhimani@gmail.com](mailto:kevinhbhimani@gmail.com)
  - For: Technical queries, bug reports, and development contributions.

- **Aobo Li**
  - Email: [aol002@ucsd.edu](mailto:aol002@ucsd.edu)
  - For: General inquiries, research collaboration, and project insights.

