# CPU-Net
Cyclic Positional U-Net (CPU-Net) is a transfer learning model for ad-hoc pulse shape translation in HPGe detectors.
- This is the source code of training and validating the performance of CPU-Net
    - GPU: Nvidia A100 Node
    - RAM usage: About 6Gb
    - Training Time: 20 - 30min
- To run this code, open `TrainAndPlot.ipynb`
- This code should run on a standard python environment with PyTorch installed.
    - PyTorch Version used: 1.9.0, but should be compatible with earlier version as well
    - we recommend installing a small gadget `tqdm` to monitor the time/progress of `for` loops. Installation can be done with `pip install tqdm --user`
    - If the user do not wish to install `tqdm`, please delete the import code and tdqm() wrapper
- This repository only contains the script of the model, training data has to be downloaded separately at [here](https://drive.google.com/file/d/1JcgQy6snavgcRetFAGl0QM3OAmPTqKqt/view?usp=sharing).
- Once downloaded, please unzip it and dump it into the same folder with `TrainAndPlot.ipynb`
# Training Result
## Network translation performance
![alt text](https://github.com/aobol/CPU-Net/blob/3061aba77858266237940826869d0b5a332aced1/ATN.png?raw=True)
## Distribution of Normalized Tail Slope
![alt text](https://github.com/aobol/CPU-Net/blob/3061aba77858266237940826869d0b5a332aced1/tailslope.png)
## Distribution of Current Amplitude
![alt text](https://github.com/aobol/CPU-Net/blob/3061aba77858266237940826869d0b5a332aced1/current_amp.png)
