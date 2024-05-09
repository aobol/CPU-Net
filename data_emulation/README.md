# Signal Modeling and Simulation Readme

## Introduction
This repository contains code for modeling electronic response and noise in signals obtained from High Purity Germanium (HPGe) detectors. The library of charge signals throughout the detector is generated using the icpc_siggen source code for symmetric HPGe detectors using detector specifications of your choosing. The code utilizes real data of Double Escape Peak (DEP) and Full Energy Peak (FEP) waveforms, scales their amplitudes to normalized waveforms, applies a single pole transfer function to model the electronic readout system, and adds randomized pink noise to simulate the real-world scenario. 

## Requirements
- Python 3.x
- NumPy
- Matplotlib
- Pandas
- SciPy
- icpc_siggen repository (separately required)

## Code Structure
- **CreatingSignalLibrary.ipynb**: Jupyter notebook used to create a library of simulated charge signals.
- **SimulatingElectronicResponse.ipynb**: Jupyter notebook used to add to the library with waveforms complete with background and noise
- **Normal_Library/**: Directory to store the generated library of charge signals.
- **dep_waveforms.npz**: Data file containing DEP waveforms.
- **fep_waveforms.npz**: Data file containing FEP waveforms.
- **config_files/**: Directory containing configuration files for the simulation.
- **fields/**: Directory containing field data files.
- **my_stester_commands.txt**: Text file containing commands for stester.

## Usage
1. **Generate Charge Signal Library**:
   - Utilize the icpc_siggen source code in a separate repository to generate a library of charge signals throughout the detector.
   - Run simulations for various positions within the detector and store the library in a pickle file (`output_dataframe.pkl`).

2. **Plot Charge Signals**:
   - Visualize charge signals at specific positions within the detector using `plot_position()` and `plotwaveform()` functions.
   - Utilize `plotwaveform()` function to plot charge signals for different combinations of radial and axial positions.

3. **Load and Analyze Waveforms**:
   - Load DEP and FEP waveforms from respective data files.
   - Scale amplitudes to normalized waveforms and calculate sigma values.
   - Apply transfer function to waveforms to model electronic readout system.
   - Add pink noise to simulate realistic noise conditions.

## Additional Notes
- Ensure the existence of a folder named **Normal_Library** in the directory to successfully run the simulation for creating the signal library.
- Modify configuration files in **config_files/** to customize simulation parameters.
- Execute the provided code snippets sequentially for desired analysis and visualization of charge signals.
