import numpy as np
import torch.utils.data as data_utils
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
'''
Parameters for training waveform construction.
LSPAN: how many sample to select to the left of time point 0 (start of the rise)
RSPAN: how many sample to select to the right of time point 0 (start of the rise)
SEQ_LEN: total length of the input pulses, always equal to LSPAN+RSPAN
'''
LSPAN=300
RSPAN=500
SEQ_LEN=LSPAN+RSPAN

class SplinterDataset(Dataset):
    '''
    Splinter is the name of our local Ge detector
    '''

    def __init__(self, event_dset = "DetectorPulses.pickle", siggen_dset ="SimulatedPulses.pickle"):
        
        event_dict = self.event_loader(event_dset)
        siggen_dict = self.event_loader(siggen_dset)
        
        self.siggen_dict = siggen_dict
        self.event_dict = event_dict
        self.size = len( self.event_dict)
        self.sim_size = len( self.siggen_dict)
        print(self.size)

        self.plot_waveform(np.random.randint(self.size))
        
        
        
        
    def __len__(self):
        return self.size
    
    
    def transform(self,wf, tp0, MC=False):
        wf = np.array(wf)
        try:
            tp50=tp0[0]
        except:
            tp50 = tp0
        left_padding = max(LSPAN-tp50,0)
        right_padding = max((RSPAN+tp50)-len(wf),0)
        wf = np.pad(wf,(left_padding, right_padding),mode='edge')
        tp50 = tp50+left_padding
        wf = wf[(tp50-LSPAN):(tp50+RSPAN)]
        wf = (wf - wf.min())/(wf.max()-wf.min())
        return wf
            
        

    # @torchsnooper.snoop()
    def __getitem__(self, idx):
        #stack two waveforms together randomly
        # np.random.seed(idx)
        siggendict1 = self.siggen_dict[np.random.randint(self.sim_size)]
        siggendict2 = self.siggen_dict[np.random.randint(self.sim_size)]
        randflag = np.random.rand()
        # if randflag > 0.7:
        #     alpha = 1
        # elif randflag < 0.1:
        #     alpha = 511/(2615-511)
        # else:
        #     alpha = np.random.rand()
        alpha = 511/(2614.5-511)
        if randflag > 0.3:
            alpha = 1
        # elif randflag > 0.2:
        #     alpha = np.random.rand()
        siggenwf1 = self.transform(siggendict1["wf"],siggendict1["tp0"],MC=True)
        siggenwf2 = self.transform(siggendict2["wf"],siggendict2["tp0"],MC=True)
        siggenwf = (siggenwf1*alpha+siggenwf2*(1-alpha))
        
        return self.transform(self.event_dict[idx]["wf"],self.event_dict[idx]["tp0"])[None,:], siggenwf[None,:], self.event_dict[idx]["wf"][None,:SEQ_LEN]
        
    def return_label(self):
        return self.trainY
    
    def set_raw_waveform(self,raw_wf):
        self.raw_waveform = raw_wf

    def get_original_waveform(self,wf, input=False):
        if input:
            return self.input_transform.recon_waveform(wf)
        else:
            return self.output_transform.recon_waveform(wf)
    
    #Load event from .pickle file
    def event_loader(self, address,elow=-99999,ehi=99999):
        wf_list = []
        ts_list = []
        count = 0
        with (open(address, "rb")) as openfile:
            while True:
                try:
                    wdict = pickle.load(openfile, encoding='latin1')
                    wf = wdict["wf"]
                    if "dc_label" in wdict.keys() and wdict["dc_label"] != 0.0:
                        continue
                    tp0 = wdict["tp0"]
                    try:
                        tp0=tp0[0]
                    except:
                        tp0 = tp0
                    nwf = (wf - wf.min())/(wf.max()-wf.min())
                    if np.nan in nwf:
                        continue
                    # if (self.pileup_cut(nwf)>7):
                    #     continue
                    # plt.plot(nwf[tp0:tp0+100])
                    if len(self.transform(wdict["wf"],wdict["tp0"],MC=True)) == SEQ_LEN:
                        wf_list.append(wdict)
                        count += 1
                except EOFError:
                    break
        return wf_list
    
    def get_field_from_dict(self, input_dict, fieldname):
        field_list = []
        for event in input_dict:
            field_list.append(event[fieldname])
        return field_list
    
    def get_current_amp(self,wf):
        return max(np.diff(wf.flatten()))
    
    def plot_waveform(self,idx):
        plt.figure(figsize=(15,15))
        plt.subplot(211)
        for i in range(100):
            waveform, waveform_deconv, rawwf = self.__getitem__(i)
            plt.plot(waveform[0],linewidth=0.5)
        plt.title("Smoothed Data")
        plt.xlabel("Time Sample")
        plt.ylabel("ADC counts")
        plt.subplot(212)
        for i in range(100):
            waveform, waveform_deconv, rawwf = self.__getitem__(i)
            plt.plot(waveform_deconv[0],linewidth=0.5)
        plt.title("Simulated WF")
        plt.xlabel("Time Sample")
        plt.ylabel("ADC counts")