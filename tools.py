import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.autograd as autograd

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


def get_tail_slope(wf):
    '''
    This function calculates the tail slope of input waveform
    '''
    premax_wf = wf[:wf.argmax()]
    point97 = np.argmin(np.abs(premax_wf - 0.97))
    last_pt = point97+200
    first_occurence = np.mean(wf[(last_pt-50):(last_pt)])
    last_occurence = np.mean(wf[-100:-50])
    return (last_occurence-first_occurence)/(len(wf)-50-last_pt)


def get_ca(wf):
    '''
    This function calculates the current amplitude of input waveform
    using a sliding window linear fit
    '''
    window = 10
    dtslope = []
    for cur_index in range(len(wf)-10):
        x = np.arange(cur_index,cur_index+window,1)
        y = wf[cur_index:cur_index+window]
        # # print(x.shape,y.shape)
        dtslope.append(np.polyfit(x,y,1)[0])
        # dtslope.append((wf[cur_index+window]-wf[cur_index])/10)
    return np.max(dtslope)

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
    quantilelow = np.quantile(x,0.10)
    quantilehi = np.quantile(x,0.90)
    return x[(x>quantilelow) & (x<quantilehi)]

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
        for wf, wf_deconv,rawwf in train_loader:
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