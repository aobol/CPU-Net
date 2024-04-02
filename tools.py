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
    waveform = waveform/1000
    if plot:
        plt.figure(figsize=(14, 10))  # Increased figure size
        
        plt.subplot(5, 1, 1)  # Added subplot for initial waveform
        plt.plot(waveform, label="Initial Waveform")
        plt.title("Initial Waveform")
        plt.legend()

    # Step 1: Calculate the current waveform
    current = np.zeros(len(waveform) - 1)
    avg_current(waveform, 1, current)
    
    # Plot the current waveform
    if plot:
        plt.subplot(5, 1, 2)
        plt.plot(current, label="Current Waveform")
        plt.title("Step 1: Current Waveform")
        plt.legend()
    
    # Step 2: Upsample the current waveform
    upsample_factor = 16
    upsampled_current = np.zeros((len(current) - 1) * upsample_factor)
    upsampler(current, upsample_factor, upsampled_current)
    
    # Plot the upsampled current waveform
    if plot:
        plt.subplot(5, 1, 3)
        plt.plot(upsampled_current, label="Upsampled Current Waveform")
        plt.title("Step 2: Upsampled Current Waveform")
        plt.legend()
    
    # Step 3: Apply moving window to the upsampled current
    window_length = 48
    num_mw = 3
    mw_type = 0  # Alternate moving windows right and left
    smoothed_current = np.zeros_like(upsampled_current)
    moving_window_multi(upsampled_current, window_length, num_mw, mw_type, smoothed_current)
    
    # Plot the smoothed current waveform
    if plot:
        plt.subplot(5, 1, 4)
        plt.plot(smoothed_current, label="Smoothed Current Waveform")
        plt.title("Step 3: Smoothed Current Waveform")
        plt.legend()
    
    # Step 4: Find A-Max in the smoothed current waveform
    t_min, t_max, A_min, A_max = np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1)
    min_max(smoothed_current, t_min, t_max, A_min, A_max)
    
    # Plot the final waveform highlighting A-Max
    if plot:
        plt.subplot(5, 1, 5)
        plt.plot(smoothed_current, label="Final Smoothed Current Waveform")
        plt.scatter(t_max, A_max, color='red', label="A-Max")
        plt.title("Step 4: Final Smoothed Current Waveform with A-Max")
        plt.legend()
        plt.tight_layout()  # Adjust layout to make sure everything fits
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


# def get_tail_slope(wf):
#     '''
#     This function calculates the tail slope of input waveform
#     '''
#     premax_wf = wf[:wf.argmax()]
#     point97 = np.argmin(np.abs(premax_wf - 0.97))
#     last_pt = point97+200
#     first_occurence = np.mean(wf[(last_pt-50):(last_pt)])
#     last_occurence = np.mean(wf[-100:-50])
#     return (last_occurence-first_occurence)/(len(wf)-50-last_pt)


# def get_tail_slope(wf):
#     '''
#     This function calculates the tail slope of input waveform.
#     Skips the calculation if the waveform contains NaN values.
#     '''
#     # Check if the waveform contains any NaN values
#     if np.isnan(wf).any():
#         # Return NaN or some default value to indicate the issue
#         return 0

#     premax_wf = wf[:wf.argmax()]
#     point97 = np.argmin(np.abs(premax_wf - 0.97))
#     last_pt = point97 + 200
#     # Ensure last_pt does not exceed the length of wf
#     last_pt = min(last_pt, len(wf) - 1)
    
#     first_occurence = np.mean(wf[(last_pt-50):(last_pt)])
#     last_occurence = np.mean(wf[-100:-50])
#     slope = (last_occurence - first_occurence) / (len(wf) - 50 - last_pt)

#     return slope

def linear(x, a, b):
        """Linear function ax + b"""
        return a * x + b
    
def get_tail_slope(wf, plot_sample=False):
    sample = 300
    fit_coefficients = []
    if len(wf) < sample:
        return 0  # Skip waveforms with fewer than 300 samples
    x_data = np.arange(sample)
    y_data = np.log(np.clip(wf[-sample:], 1e-10, None))  # Log of last 300 samples, avoiding log(0)
    try:
        popt, pcov = curve_fit(linear, x_data, y_data, maxfev=100000)
        if plot_sample:
            plot_slope_calc(x_data, y_data, popt, wf[-sample:])
    except Exception as e:
        print(f"Failed to fit waveform due to {e}. Appending NaN values.")
        return 0
    return popt[0]

def plot_slope_calc(x_data, y_data, popt, original_wf):
    plt.figure(figsize=(12, 6))

    # Plotting the original waveform
    plt.subplot(2, 1, 1)
    plt.plot(original_wf, 'b-', label='Original Waveform')
    plt.title('Original Waveform')
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.legend()

    # Plotting the logarithm of the waveform and the linear fit
    plt.subplot(2, 1, 2)
    plt.plot(x_data, y_data, 'ro', label='Logarithm of Waveform', markersize=4)
    plt.plot(x_data, linear(x_data, *popt), 'g-', label='Linear Fit')
    plt.title('Logarithm of Waveform and Linear Fit')
    plt.xlabel('Sample')
    plt.ylabel('Log Amplitude')
    plt.legend()
    plt.tight_layout()
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
        
        
def check_peak_alignment(loader, ATN, tolerance=2):
    """
    Check the alignment of peak current amplitudes between original and translated waveforms.

    This function processes batches of waveforms from a data loader, applies an Ad-hoc Translation Network (ATN)
    to the deconvolved waveforms, calculates the current amplitude for both original and translated waveforms,
    and checks if the histogram peaks of these amplitudes align within a specified tolerance.

    Parameters:
    - loader (DataLoader): DataLoader providing batches of original and deconvolved waveforms.
    - ATN (torch.nn.Module): The Ad-hoc Translation Network model used for translating deconvolved waveforms.
    - tolerance (int, optional): The allowed difference in bin index for peak alignment check. Default is 2.

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

