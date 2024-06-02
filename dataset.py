import numpy as np
import torch.utils.data as data_utils
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from tools import calculate_tn
from tqdm import tqdm
import random
from scipy.optimize import curve_fit

'''
Parameters for training waveform construction.
LSPAN: how many sample to select to the left of time point 0 (start of the rise)
RSPAN: how many sample to select to the right of time point 0 (start of the rise)
SEQ_LEN: total length of the input pulses, always equal to LSPAN+RSPAN
'''
LSPAN=400
RSPAN=400
SEQ_LEN=LSPAN+RSPAN
t_n = 99.9
base_thres = 0.005 # mean of first 50 smaples should be less than this value
tail_thres = 0.80 # last 50 samples should be greater than this value
# chi_squared_threshold= 0.002
# popt_threshold = -2.6e-4
norm_tail_height = 0.80
norm_samples = 5
class SplinterDataset(Dataset):
    '''
    Splinter is the name of our local Ge detector
    '''
    def __init__(self, event_dset="DetectorPulses.pickle", siggen_dset="SimulatedPulses.pickle", n_max = 1e5, chi_squared_threshold=1, popt_threshold_under=-2, popt_threshold_over=2):
        self.n_max = n_max
        self.chi_squared_threshold = chi_squared_threshold
        self.popt_threshold_over = popt_threshold_over
        self.popt_threshold_under = popt_threshold_under
        self.chi_squared_coeff = []
        self.tau_fits = []
        self.event_dict, self.rejected_wf = self.event_loader_data(event_dset)
        print("Number of Data events:", len(self.event_dict))
        self.siggen_dict = self.event_loader_sim(siggen_dset)
        print("Number of Simulations events", len(self.siggen_dict))
        # Set the class attributes for thresholds here
        self.size = min(len(self.event_dict),len(self.siggen_dict))
        self.event_ids = [wdict["event"] for wdict in self.siggen_dict]
        self.plot_waveform() 
        
    def __len__(self):
        # Return the minimum size between event_dict and siggen_dict to avoid out-of-range errors
        return min(len(self.event_dict), len(self.siggen_dict))

    def __getitem__(self, idx):
        # Use a single simulated waveform based on the index and transform it
        siggenwf = self.transform(self.siggen_dict[idx]["wf"], self.siggen_dict[idx]["tp0"], sim=True)
        # Transform the real waveform for comparison or any other purpose
        real_wf = self.transform(self.event_dict[idx]["wf"], self.event_dict[idx]["tp0"])
        # Return the real waveform, the single transformed simulated waveform, and the original waveform
        event_id = self.siggen_dict[idx].get("event", -1)  # Default to -1 or suitable value if not found
        # Return the event_id as part of the output
        return real_wf[None, :], siggenwf[None, :], self.event_dict[idx], self.event_dict[idx]
        
    def return_label(self):
        return self.trainY
    
    def set_raw_waveform(self,raw_wf):
        self.raw_waveform = raw_wf

    def get_original_waveform(self,wf, input=False):
        if input:
            return self.input_transform.recon_waveform(wf)
        else:
            return self.output_transform.recon_waveform(wf)
    

    def normalize_waveform(self, wf):
        """Normalize waveform by dividing by the average of the last norm_samples samples and shifting the waveform 
           so that the average of the first 200 samples is zero."""
        tail_mean = np.mean(wf[-norm_samples:])  # Define the tail region as the last 5 samples
        if tail_mean != 0:
            normalized_wf = wf * norm_tail_height / tail_mean
        else:
            normalized_wf = wf  # If the tail mean is zero, return the waveform as is to avoid division by zero

        # Shift the waveform so that the average of the first 200 samples is zero
        first_200_mean = np.mean(normalized_wf[:200])
        normalized_wf = normalized_wf - first_200_mean

        return normalized_wf


    def normalize_sim_waveform(self, wf):
        """Normalize waveform to have values between 0 and 1."""
        min_val = np.min(wf)
        max_val = np.max(wf)
        if max_val > min_val:
            return (wf - min_val) / (max_val - min_val)
        else:
            # Handle the case where max_val equals min_val (e.g., constant waveforms)
            return np.zeros_like(wf)  # or wf * 0 to return a waveform of zeros

    def transform(self, wf, tp0, sim=False):
        """Transform waveform by padding based on tp0 and then normalizing."""
        wf = np.array(wf)
        # Ensure tp0 is an integer
        tp0 = int(round(tp0))
        left_padding = max(LSPAN - tp0, 0)
        right_padding = max((RSPAN + tp0) - len(wf), 0)
        # Apply padding
        wf_padded = np.pad(wf, (left_padding, right_padding), mode='edge')
        # Adjust tp0 after padding
        tp0_adjusted = tp0 + left_padding
        # Slice the waveform around the adjusted tp0 to ensure consistent length
        wf_sliced = wf_padded[(tp0_adjusted - LSPAN):(tp0_adjusted + RSPAN)]
        # Normalize the waveform after padding and slicing
        wf_normalized = self.normalize_waveform(wf_sliced)
        # Don't normalize if it is sim as it is already normalized
        if sim:
            return self.normalize_sim_waveform(wf_sliced)
        return wf_normalized

    def event_loader_sim(self, address, elow=-99999, ehi=99999):
        wf_list = []
        count=0
        with (open(address, "rb")) as openfile:
            while True:
                if count >self.n_max:
                    break
                try:
                    wdict = pickle.load(openfile, encoding='latin1')
                    wf = wdict["wf"]
                    if "dc_label" in wdict.keys() and wdict["dc_label"] != 0.0:
                        continue
                    # Calculate tp0 using calculate_t90 or any other method without normalizing here
                    try:
                        tp0 = calculate_tn(wf,t_n)  # Assuming calculate_t90 returns an appropriate tp0 value
                    except Exception as e:
                        continue  # Skip this waveform if tp0 cannot be calculated
                    # Store tp0 in the waveform dictionary for later transformation
                    wdict["tp0"] = tp0
                    transformed_wf = self.transform(wf, tp0)
                    if len(transformed_wf) == SEQ_LEN:
                        if np.all(~np.isnan(transformed_wf)) and np.any(transformed_wf != 0):
                            wf_list.append(wdict)
                            count+=1
                    if count % 10000 == 0:
                        print(f"{count} waveforms loaded from simulations.")
                except EOFError:
                    break
        return wf_list
    
    def event_loader_data(self, address, elow=-99999, ehi=99999):
        wf_list = []
        wf_list_rejected = []
        count=0
        print("Chi squared cut is",self.chi_squared_threshold)
        print("Tail slope cut over is", self.popt_threshold_over)
        print("Tail slope cut under is", self.popt_threshold_under)
        with (open(address, "rb")) as openfile:
            while True:
                if count >self.n_max:
                    break
                try:
                    wdict = pickle.load(openfile, encoding='latin1')
                    wf = wdict["wf"]
                    if "dc_label" in wdict.keys() and wdict["dc_label"] != 0.0:
                        continue
                    try:
                        tp0 = calculate_tn(wf,t_n)
                    except Exception as e:
                        continue  # Skip this waveform if tp0 cannot be calculated
                    # Store tp0 in the waveform dictionary for later transformation
                    wdict["tp0"] = tp0
                    transformed_wf = self.transform(wf, tp0)
                    # Skip waveforms if the transformed waveform between samples 400 to 500 is less than 0.9
                    if np.any(transformed_wf[400:420] < 0.92):
                        wf_list_rejected.append(transformed_wf)
                        continue
                    if len(transformed_wf) == SEQ_LEN:
                        # Calculate the mean of the first 250 samples
                        mean_first_250 = np.mean(transformed_wf[:250])
                        #Skip waveforms with any value > 0.01 in the first 250 samples
                        if np.any(np.array(transformed_wf[:250]) > 0.01):
                            wf_list_rejected.append(transformed_wf)
                            continue
                        #Skip waveforms with any value < tail_thres in the last 50 samples
                        if np.any(np.array(transformed_wf[-50:]) < tail_thres):
                            wf_list_rejected.append(transformed_wf)
                            continue
                        # Check if the first 100 samples' mean is <= 0.1 AND last 50 samples' mean is > 0.5
                        if mean_first_250 <= base_thres:
                            chi_squared, popt = self.process_wf_log_linear(transformed_wf)
                            if chi_squared < self.chi_squared_threshold and popt >self.popt_threshold_under and popt < self.popt_threshold_over:
                                # print('Calc chi squared', chi_squared)
                                # print('Calc popt_threshold', popt)
                                if np.all(~np.isnan(transformed_wf)) and np.any(transformed_wf != 0):
                                    # if np.max(transformed_wf)>=0.93: #this cleans the ORNL data that has bumps on top, remove for other data
                                    #     continue
                                    self.chi_squared_coeff.append(chi_squared)
                                    self.tau_fits.append(popt)
                                    wf_list.append(wdict)
                                    count+=1
                            else:
                                wf_list_rejected.append(transformed_wf)
                        else:
                            wf_list_rejected.append(transformed_wf)
                        if count % 10000 == 0:
                            print(f"{count} waveforms loaded from data.")
                except EOFError:
                    break
        return wf_list, wf_list_rejected
    
    def get_field_from_dict(self, input_dict, fieldname):
        field_list = []
        for event in input_dict:
            field_list.append(event[fieldname])
        return field_list
    
    def get_current_amp(self,wf):
        return max(np.diff(wf.flatten()))
    
    def plot_waveform(self):
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))  # Create a figure with two subplots side by side

        # Plotting 100 Random Pulses
        # axs[0].axhline(0.965)

        for i in range(1000):
            waveform, waveform_deconv, rawwf, _ = self.__getitem__(i)
            axs[0].plot(waveform[0], linewidth=0.5)

        axs[0].set_title("200 Random Data Pulses")
        axs[0].set_xlabel("Time Sample [ns]")
        axs[0].set_ylabel("Normalized Pulses")
        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[0].minorticks_on()
        axs[0].grid(which='minor', linestyle=':', linewidth='0.5')

        # Plotting 100 Simulated WF
        for i in range(200):
            waveform, waveform_deconv, rawwf, _ = self.__getitem__(i)
            axs[1].plot(waveform_deconv[0], linewidth=0.5)
        axs[1].set_title("200 Random Simulated Pulses")
        axs[1].set_xlabel("Time Sample [ns]")
        axs[1].set_ylabel("Normalized Pulses")
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)
        axs[1].minorticks_on()
        axs[1].grid(which='minor', linestyle=':', linewidth='0.5')
        plt.savefig('figs/inputs.png',dpi=200)
    
    def linear(self, x, a, b):
        """Linear function ax + b"""
        return a * x + b
    
    def process_wf_log_linear(self, wf):
        sample = 300
        if len(wf) < sample:
            # Return default values if waveform is too short
            return np.nan, [np.nan, np.nan]  # Ensure popt is a list or array to safely index [0] later
        x_data = np.arange(sample)
        y_data = np.log(np.clip(wf[-sample:], 1e-10, None))  # Log of last 300 samples
        try:
            popt, pcov = curve_fit(self.linear, x_data, y_data, maxfev=100000)
            # Calculate residuals and chi-squared for goodness of fit
            residuals = y_data - self.linear(x_data, *popt)
            chi_squared = np.sum((residuals ** 2) / self.linear(x_data, *popt))
        except Exception as e:
            # Handle fitting errors
            popt = [np.nan, np.nan]  # Ensure popt is a list or array
            chi_squared = np.nan
        return -chi_squared, popt[0] #chi squared would be negative since log of number between 0,1 is negavtive, so we return positive value  
    