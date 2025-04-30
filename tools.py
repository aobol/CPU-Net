import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.autograd as autograd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
import os
from tqdm import tqdm
from dspeed.processors import avg_current, upsampler, moving_window_multi, min_max, time_point_thresh
from dspeed.errors import DSPFatal

from dspeed.processors import avg_current, upsampler, moving_window_multi, min_max, time_point_thresh

def calc_current_amplitude(waveform, plot=False):
    """
    Process a waveform through the specified DSP chain, with optional plotting.
    
    Parameters:
    waveform : numpy.ndarray
        The waveform to process.
    plot : bool
        Whether to plot the intermediate steps.
    
    Returns:
    A_max : float
        The maximum amplitude of the current waveform after processing.
    """
    waveform = waveform
    if plot:
        plt.figure(figsize=(14, 10))  # Increased figure size
        
        plt.subplot(5, 1, 1)  # Added subplot for initial waveform
        plt.plot(waveform)
        plt.title("Initial Waveform", fontsize=14)

    # Step 1: Calculate the current waveform
    current = np.zeros(len(waveform) - 1)
    avg_current(waveform, 1, current)
    
    # Plot the current waveform
    if plot:
        plt.subplot(5, 1, 2)
        plt.plot(current)
        plt.title("Step 1: Current Waveform", fontsize=14)
    
    # Step 2: Upsample the current waveform
    upsample_factor = 16
    upsampled_current = np.zeros((len(current) - 1) * upsample_factor)
    upsampler(current, upsample_factor, upsampled_current)
    
    # Plot the upsampled current waveform
    if plot:
        plt.subplot(5, 1, 3)
        plt.plot(upsampled_current)
        plt.title("Step 2: Upsampled Current Waveform", fontsize=14)
    
    # Step 3: Apply moving window to the upsampled current
    window_length = 48
    num_mw = 3
    mw_type = 0  # Alternate moving windows right and left
    smoothed_current = np.zeros_like(upsampled_current)
    moving_window_multi(upsampled_current, window_length, num_mw, mw_type, smoothed_current)
    
    # Plot the smoothed current waveform
    if plot:
        plt.subplot(5, 1, 4)
        plt.plot(smoothed_current)
        plt.title("Step 3: Smoothed Current Waveform", fontsize=14)
    
    # Step 4: Find A-Max in the smoothed current waveform
    t_min, t_max, A_min, A_max = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    min_max(smoothed_current, t_min, t_max, A_min, A_max)
    
    # Plot the final waveform highlighting A-Max
    if plot:
        plt.subplot(5, 1, 5)
        plt.plot(smoothed_current)
        plt.scatter(t_max, A_max, color='red', label="A-Max")
        plt.title("Step 4: Smoothed Current Waveform with A-Max", fontsize=14)
        plt.legend(fontsize=14)
        plt.tight_layout()  # Adjust layout to make sure everything fits
        plt.savefig('figs/curr_amp_calc.pdf')
        plt.show()
    
    return A_max[0]

def process_all_waveforms(directory):
    A_max_values = []
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".txt"):
            waveform_path = os.path.join(directory, filename)
            waveform = np.loadtxt(waveform_path)
            A_max = calc_current_amplitude(waveform)
            A_max_values.append(A_max)
    return A_max_values


def calculate_tn(wf, n=90):
    """
    Calculate the time point of the maximum amplitude (tp_max) and the time
    when the waveform reaches n% of its maximum amplitude (tn).

    Parameters
    ----------
    wf : np.ndarray
        The waveform array, assumed to be baseline-subtracted and pole-zero corrected.

    Returns
    -------
    t90 : float
        The time point (in samples) where the waveform reaches n% of its maximum amplitude.
    tp_max : int
        The index of the maximum amplitude in the waveform.
    """
    # Ensure wf is a numpy array
    wf = np.asarray(wf)

    # Find the maximum amplitude and its index
    max_amplitude = np.max(wf)
    tp_max = np.argmax(wf)

    # Calculate n% of the maximum amplitude
    
    threshold_n = n/100 * max_amplitude

    # Initialize t90 as NaN
    tn = np.nan

    # Search for the crossing point
    for i in range(len(wf)):
        if wf[i] >= threshold_n:
            tn = i
            break
    return tn

def asym_trap_filter(w_in, rise, flat, fall):
    """
    Apply an asymmetric trapezoidal filter to the waveform, normalized
    by the number of samples averaged in the rise and fall sections.
    Parameters
    ----------
    w_in : array-like
        The input waveform
    rise : int
        The number of samples averaged in the rise section
    flat : int
        The delay between the rise and fall sections
    fall : int
        The number of samples averaged in the fall section
    w_out : array-like
        The normalized, filtered waveform
    Examples
    --------
    .. code-block :: json
        "wf_af": {
            "function": "asym_trap_filter",
            "module": "pygama.dsp.processors",
            "args": ["wf_pz", "128*ns", "64*ns", "2*us", "wf_af"],
            "unit": "ADC",
            "prereqs": ["wf_pz"]
        }
    """
    w_out = np.array([np.nan]*len(w_in))
    w_in = (w_in-w_in.min())/(w_in.max()-w_in.min())
    if np.isnan(w_in).any() or np.isnan(rise) or np.isnan(flat) or np.isnan(fall):
        return

    w_out[0] = w_in[0] / rise
    for i in range(1, rise, 1):
        w_out[i] = w_out[i-1] + w_in[i] / rise
    for i in range(rise, rise + flat, 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise]) / rise
    for i in range(rise + flat, rise + flat + fall, 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise]) / rise - w_in[i-rise-flat] / fall
    for i in range(rise + flat + fall, len(w_in), 1):
        w_out[i] = w_out[i-1] + (w_in[i] - w_in[i-rise]) / rise - (w_in[i-rise-flat] - w_in[i-rise-flat-fall]) / fall
    return w_out

def get_roc(sig,bkg):
    '''
    This function gets the false positive rate, true positive rate, cutting threshold 
    and area under curve using the given signal and background array
    '''
    testY = np.array([1]*len(sig) + [0]*len(bkg))
    predY = np.array(sig+bkg)
    auc = roc_auc_score(testY, predY)
    fpr, tpr, thr = roc_curve(testY, predY)
    return fpr,tpr,thr,auc

def linear(x, a, b):
        """Linear function ax + b"""
        return a * x + b
    
def get_tail_slope(wf, plot_sample=False, return_uncertainty=False):
    sample = 300
    if len(wf) < sample:
        return (0, 0) if return_uncertainty else 0
    x_data = np.arange(sample)
    y_data = np.log(np.clip(wf[-sample:], 1e-10, None))  # Avoid log(0)
    try:
        popt, pcov = curve_fit(linear, x_data, y_data, maxfev=100000)
        if plot_sample:
            plot_slope_calc(x_data, y_data, popt, wf[-sample:])
        slope = popt[0]
        slope_unc = np.sqrt(pcov[0, 0])
    except Exception as e:
        print(f"Failed to fit waveform due to {e}.")
        slope = 0
        slope_unc = 0
    return (slope, slope_unc) if return_uncertainty else slope


def plot_slope_calc(x_data, y_data, popt, original_wf):
    plt.figure(figsize=(12, 6))

    # Plotting the original waveform
    plt.subplot(2, 1, 1)
    plt.plot(original_wf, 'b-')
    plt.title('Original Waveform',fontsize=14)
    plt.xlabel('Sample',fontsize=14)
    plt.ylabel('Normalized Signal',fontsize=14)

    # Plotting the logarithm of the waveform and the linear fit
    plt.subplot(2, 1, 2)
    plt.plot(x_data, y_data, 'ro', label='Logarithm of Waveform', markersize=4)
    plt.plot(x_data, linear(x_data, *popt), 'g-', label='Linear Fit')
    plt.title('Logarithm of Waveform and Linear Fit',fontsize=14)
    plt.xlabel('Sample',fontsize=14)
    plt.ylabel('Log Signal',fontsize=14)
    plt.legend()
    plt.xticks(fontsize=12)  # Bigger tick labels
    plt.yticks(fontsize=12)  # Bigger tick labels
    plt.tight_layout()
    plt.savefig('figs/tail_slope_calc.pdf', dpi=100)
    plt.show()  
                
            
def calc_gradient_penalty(netD, real_data, fake_data):
    '''
    This function calculates the gradient penalty of GAN-based model (ArXiv: 1704.00028)
    The idea is to apply 1-Lipshitz constratin on the latent space
    '''
    alpha = torch.rand(BATCH_SIZE, 1,1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.to(DEVICE)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(DEVICE)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(DEVICE),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()*10
    return gradient_penalty


def select_quantile(x):
    '''
    Select only the 10% to 90% quantile of the data
    used to calculate a more robust mean/std of long-tailed dataset
    '''
    x = np.array(x)  # Ensure x is a NumPy array
    quantilelow = np.quantile(x, 0.10)
    quantilehi = np.quantile(x, 0.90)
    return x[(x > quantilelow) & (x < quantilehi)]

def calculate_iou(h1,h2, rg, normed=False):
    '''
    Calculate the histogram intersection over union
    '''
    h1 = np.array(h1)
    h2 = np.array(h2)
    if normed:
        mean,std = norm.fit(select_quantile(h1))
        h1 = (h1-mean)/std
        mean,std = norm.fit(select_quantile(h2))
        h2 = (h2-mean)/std
    count, _ = np.histogram(h1,bins=rg,density=True)
    count2, _ = np.histogram(h2,bins=rg,density=True)
    intersection = 0
    union = 0
    for i in range(len(count)):
        intersection += min(count[i],count2[i])
        union += max(count[i],count2[i])
    return intersection/union*100.0

def inf_train_gen(train_loader):
    '''
    Allow us to sample infinitely (with repetition) from the training dataset
    '''
    while True:
        for wf, wf_deconv,rawwf,_ in train_loader:
            yield wf, wf_deconv
            
class LambdaLR():
    '''
    Controls the learning rate decay
    '''
    def __init__(self, n_epochs, offset, decay_start_epoch):
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)

def weights_init_normal(m):
    '''
    Weight initialization
    '''
    classname = m.__class__.__name__
    dev  = 0.02
    if classname.find('Conv1d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, dev)
    if classname.find('ConvTranspose1d') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, dev)
    elif classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, dev)
    elif classname.find('BatchNorm1d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, dev)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
        
def check_peak_alignment(loader, ATN, tolerance, DEVICE):
    """
    Check the alignment of peak current amplitudes between original and translated waveforms.

    This function processes batches of waveforms from a data loader, applies an Ad-hoc Translation Network (ATN)
    to the deconvolved waveforms, calculates the current amplitude for both original and translated waveforms,
    and checks if the histogram peaks of these amplitudes align within a specified tolerance.

    Parameters:
    - loader (DataLoader): DataLoader providing batches of original and deconvolved waveforms.
    - ATN (torch.nn.Module): The Ad-hoc Translation Network model used for translating deconvolved waveforms.
    - tolerance (int, optional): The allowed difference in bin index for peak alignment check.

    Returns:
    - A tuple (bool, float, float) containing:
      - A boolean value indicating if the peak alignment is within the specified tolerance.
      - The peak location of current amplitudes for original waveforms.
      - The peak location of current amplitudes for translated waveforms.
    """   
    ca = []
    gan_ca = []
    rg = np.linspace(0.00009, 0.00015, 70)
    i=0 
    for wf, wf_deconv, a, b in loader:
        if i==30: #how many batches to process
            break
        bsize = wf.size(0)
        gan_wf = ATN(wf_deconv.to(DEVICE).float())
        for iwf in range(bsize):
            datawf = wf[iwf,0].cpu().numpy().flatten()
            transfer_wf = gan_wf[iwf,0].detach().cpu().numpy().flatten()
            ca.append(calc_current_amplitude(datawf))
            gan_ca.append(calc_current_amplitude(transfer_wf))
        i += 1 
    counts_ca, bin_edges_ca = np.histogram(ca, bins=rg)
    counts_gan_ca, _ = np.histogram(gan_ca, bins=rg)
    max_count_index_ca = np.argmax(counts_ca)
    max_count_index_gan_ca = np.argmax(counts_gan_ca)
    # Calculate the peak locations
    peak_location_ca = (bin_edges_ca[max_count_index_ca] + bin_edges_ca[max_count_index_ca + 1]) / 2
    peak_location_gan_ca = (bin_edges_ca[max_count_index_gan_ca] + bin_edges_ca[max_count_index_gan_ca + 1]) / 2
    # Check if the bins with the highest frequency are within the specified tolerance
    return abs(max_count_index_ca - max_count_index_gan_ca) <= tolerance, peak_location_ca, peak_location_gan_ca

import torch

def differentiable_get_tail_slope(wf, sample=300):
    """
    Differentiable approximation of the get_tail_slope function that returns the slope (popt[0]) of the log-linear fit on the tail of the waveform.

    Parameters:
    - wf (torch.Tensor): The input waveform, expected to be a 1D tensor.
    - sample (int): The number of samples from the end of the waveform to use for slope estimation.

    Returns:
    - slope (torch.Tensor): The estimated slope of the log-linear fit (equivalent to popt[0] in the original function).
    """
    # Ensure that the waveform is a tensor
    if not isinstance(wf, torch.Tensor):
        wf = torch.tensor(wf, dtype=torch.float32)

    # If the waveform is shorter than the sample size, return 0
    if wf.size(0) < sample:
        return torch.tensor(0.0)

    # Extract the tail part of the waveform
    tail = wf[-sample:]
    
    # Avoid log(0) by clamping the values
    tail_clamped = torch.clamp(tail, min=1e-10)
    
    # Calculate the natural logarithm of the tail
    log_tail = torch.log(tail_clamped)
    
    # Create a tensor of x_data (time steps)
    x_data = torch.arange(0, sample, dtype=torch.float32, device=wf.device)
    
    # Perform linear regression to estimate the slope
    A = torch.stack([x_data, torch.ones_like(x_data)], dim=1)  # Design matrix
    solution = torch.linalg.lstsq(A, log_tail.view(-1, 1)).solution
    
    # The slope corresponds to the first element of the solution
    slope = solution[0, 0]
    
    return slope

import torch
import torch.nn.functional as F

def kl_divergence_loss(real_slopes, fake_slopes, epsilon=1e-8, scale_factor=1e6):
    """
    Computes the KL Divergence between the distributions of real and fake tail slopes.

    Parameters:
    - real_slopes (torch.Tensor): Tensor of tail slopes from the real waveforms.
    - fake_slopes (torch.Tensor): Tensor of tail slopes from the generated (fake) waveforms.
    - epsilon (float): A small value added to the standard deviation to prevent zero variance.
    - scale_factor (float): Factor by which to scale the slopes to avoid numerical issues with small values.

    Returns:
    - loss (torch.Tensor): The KL Divergence loss.
    """
    # Scale the slopes
    real_slopes_scaled = real_slopes * scale_factor
    fake_slopes_scaled = fake_slopes * scale_factor

    # Calculate the mean and standard deviation of the real and fake distributions
    real_mean = torch.mean(real_slopes_scaled)
    real_std = torch.std(real_slopes_scaled) + epsilon  # Add epsilon to avoid zero std
    
    fake_mean = torch.mean(fake_slopes_scaled)
    fake_std = torch.std(fake_slopes_scaled) + epsilon  # Add epsilon to avoid zero std
    
    # Create normal distributions based on the calculated statistics
    real_distribution = torch.distributions.Normal(real_mean, real_std)
    fake_distribution = torch.distributions.Normal(fake_mean, fake_std)
    
    # Compute the KL Divergence
    kl_loss = torch.distributions.kl_divergence(real_distribution, fake_distribution)
    
    return kl_loss

from scipy.stats import wasserstein_distance

def wasserstein_loss(real_slopes, fake_slopes, scale_factor=1.0):
    # Ensure the inputs are correctly shaped tensors
    if len(real_slopes.shape) > 1:
        real_slopes = real_slopes.flatten()
    if len(fake_slopes.shape) > 1:
        fake_slopes = fake_slopes.flatten()
    
    # Convert to numpy and scale
    real_slopes_scaled = real_slopes.detach().cpu().numpy() * scale_factor
    fake_slopes_scaled = fake_slopes.detach().cpu().numpy() * scale_factor
    
    # Ensure the arrays are 1D
    if isinstance(real_slopes_scaled, (np.ndarray, list)):
        real_slopes_scaled = np.atleast_1d(real_slopes_scaled)
    if isinstance(fake_slopes_scaled, (np.ndarray, list)):
        fake_slopes_scaled = np.atleast_1d(fake_slopes_scaled)
    
    # Check that these are indeed arrays with length
    if real_slopes_scaled.size == 0 or fake_slopes_scaled.size == 0:
        raise ValueError("Input arrays to wasserstein_distance must not be empty.")

    # Calculate the Wasserstein distance
    w_loss = wasserstein_distance(real_slopes_scaled, fake_slopes_scaled)

    # Convert the loss to a torch tensor to integrate into your loss function
    return torch.tensor(w_loss, dtype=torch.float32).to(real_slopes.device)
