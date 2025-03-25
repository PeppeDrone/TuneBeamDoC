# %% Imports
import mne
import seaborn as sns
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import OrderedDict
from IPython.display import clear_output
import sys
from scipy.io import loadmat
from termcolor import colored
import pickle
import pdb
import neurokit2 as nk
from mne.viz import plot_raw
from pyprep.prep_pipeline import PrepPipeline
import contextlib
from IPython import get_ipython
import time
from lempel_ziv_complexity import lempel_ziv_complexity as lzc
from hurst import compute_Hc, random_walk
import json
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from numpy import linalg
import scipy.signal
from mne_icalabel import label_components
from mne_icalabel.gui import label_ica_components
from autoreject import (AutoReject, get_rejection_threshold, Ransac)
from mne.datasets import (fetch_fsaverage, sample)
import os.path as op
from mne.minimum_norm import (
    make_inverse_operator, apply_inverse, apply_inverse_raw)
from mne.beamformer import (make_lcmv, apply_lcmv)
from mpl_toolkits.mplot3d import Axes3D
import h5py
from matplotlib import ticker
from scipy.io import loadmat
import h5py
from scipy.stats import mannwhitneyu
from numpy.linalg import eig, inv
from matplotlib.patches import Ellipse
from skimage.measure import EllipseModel
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
import pingouin as pg
import matplotlib.gridspec as gridspec

# %% Computation functions
def compute_fractal_analysis(data, optEX):
    fra_ = []
    print('--- Fractal analysis...')
    for ie, e in enumerate(data):
        print('-- Elaborating epoch no. ' + str(ie))
        for ixch, ec in enumerate(e):  
            eeg_normalized = (ec - np.mean(ec)) / np.std(ec)
            df, _ = nk.complexity(eeg_normalized, which = "makowski2022")
            df = df.add_suffix('_'+optEX['channels'][ixch])
            if ixch == 0:
                DF = df
            else:
                DF = pd.concat([DF,df], axis = 1)
        fra_.append(DF)            
    return pd.concat(fra_).mean()

def compute_source_metrics(epochs, evoked, optEX):
    noise_cov = mne.compute_covariance(epochs, tmin=0.3, tmax=1, method="shrunk", verbose=True)
    fs_dir = fetch_fsaverage(verbose=True)
    src_fname = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')
    bem_fname = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    src = mne.read_source_spaces(src_fname)
    fwd = mne.make_forward_solution(evoked.info, trans='fsaverage', 
                                    src=src_fname,
                                    bem=bem_fname,
                                    meg=False, eeg=True)
    inv = mne.minimum_norm.make_inverse_operator(evoked.info, fwd, 
                                                    noise_cov=noise_cov, loose=0.2)   
    snr = 3.0
    lambda2 = 1.0 / snr**2
    method = "dSPM"  
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs,
        inv,
        lambda2,
        method,
        pick_ori="normal",
        nave=evoked.nave
    )

    # Dipole peaks and stimulation-side dependance
    labels = mne.read_labels_from_annot("fsaverage", parc="aparc", subjects_dir = op.dirname(fs_dir))
    mean_stc = sum(stcs) / len(stcs)
    mean_mean_stc = np.mean(mean_stc.data, axis = 0)
    peak_vertex_ix_rx, peak_time_rx = mean_stc.get_peak(hemi="rh", vert_as_index=True,
                                                  time_as_index=False)
    dipole_pos = {}
    dipole_pos['Right position'] = src[1]['rr'][peak_vertex_ix_rx]
    dipole_pos['Right timing'] = peak_time_rx
    peak_vertex_ix_lx, peak_time_lx = mean_stc.get_peak(hemi="lh", vert_as_index=True,
                                                  time_as_index=False)
    dipole_pos['Left position'] = src[0]['rr'][peak_vertex_ix_lx]
    dipole_pos['Left timing'] = peak_time_lx
    if optEX['side'][optEX['ix']] == 1:
        hemi = 'lh'
        pos = peak_vertex_ix_lx
        labels_dict_ = {'PMC':48, #BA6
                        'PSC':44, #BA1-2-3
                        'SPL':58, #BA5-7
                        'IPL':14}
    else:
        hemi = 'rh'
        pos = peak_vertex_ix_rx
        labels_dict_ = {'PMC':49,
                        'PSC':45,
                        'SPL':59,
                        'IPL':15}
        
    stc_activity_area, stc_activity_labels = {}, {}
    for ixl, lab in enumerate(labels_dict_):
        # stc_activity_area[lab] = []
        # stc_activity_labels[lab] = mean_stc.in_label(labels[labels_dict_[lab]])
        # for t_up in [0.05,0.1,0.15,0.2,0.25]:
        #     up_to_samples = round(t_up*512)
        #     stc_activity_area[lab].append(scipy.integrate.simps(abs(stc_activity_labels[lab].data.mean(0)[0:up_to_samples]),
        #                                                             mean_stc.times[0:up_to_samples]))
        stc_activity_labels[lab] = mean_stc.in_label(labels[labels_dict_[lab]]).data.mean(axis = 0)

    
    # Compute resolution matrices and PSF
    rm_mne = mne.minimum_norm.make_inverse_resolution_matrix(fwd, inv,
                                            method="MNE", lambda2=lambda2,
                                            verbose = True)
    label_names = [label.name for label in labels]
    lh_labels = [name for name in label_names if name.endswith("lh")]
    label_ypos = list()
    for name in lh_labels:
        idx = label_names.index(name)
        ypos = np.mean(labels[idx].pos[:, 1])
        label_ypos.append(ypos)
    lh_labels = [label for (yp, label) in sorted(zip(label_ypos, lh_labels))]
    rh_labels = [label[:-2] + "rh" for label in lh_labels]
    lh_labels.remove("unknown-lh")
    rh_labels.remove("unknown-rh")
    labels = labels[:-1] 
    stcs_psf_mne = mne.minimum_norm.get_point_spread(rm_mne, src,
                                                    labels, mode="svd",
                                                    norm=None, verbose = True)
    psfs_mat = np.zeros([len(labels), stcs_psf_mne[0].shape[0]])
    for [i, s] in enumerate(stcs_psf_mne):
        psfs_mat[i, :] = s.data[:, 0]
    leakage_mne = np.abs(np.corrcoef(psfs_mat))
    
    return dipole_pos, stc_activity_labels, leakage_mne

def ParametrizePCI(ept, opt_ex):
    PCI_opt_dict = {}
    PCI_opt_dict['k'], PCI_opt_dict['min_snr'], PCI_opt_dict['pci'] = [], [], []
    times = np.linspace(0,1000,512)
    for m_snr in opt_ex['pars_snr']:
        for kk in opt_ex['pars_k']:
            print('PCI parameter: ' + 'MinSNR = ' + str(m_snr) + ', K = ' + str(kk))
            par = {'baseline_window':(500,1000),
                   'response_window':(0,500),
                   'k':kk, 'min_snr':m_snr, 
                   'max_var':99, 'embed':False, 'n_steps':10} 
            PCI_opt_dict['k'].append(kk)
            PCI_opt_dict['min_snr'].append(m_snr)
            pci_ = []
            for ie, e in enumerate(ept):
                pci_.append(calc_PCIst(e, times, **par))
            PCI_opt_dict['pci'].append(np.mean(pci_))
            
    return PCI_opt_dict     

from pathlib import Path

def process_EEG_tb(data, opt_ex, micro):
    
    for ix, dd in enumerate(data):   
        if micro:
            name_file = f"features_{ix}_nns.pkl"
        else:
            name_file = f"features_{ix}_scs.pkl"
            
        my_file = Path(os.path.join(opt_ex['res_dir'],name_file))
        if my_file.exists():
            with open(my_file, 'rb') as file:
                features = pickle.load(file)
        else:
            features = {}  

        if ix > 0: # If you want to investigate a specific patient
            opt_ex['ix'] = ix
            print(colored('Working with patient: ' + str(ix), 'red'))
            
            # Data preparation
            info = mne.create_info(ch_names=opt_ex['channels'], sfreq=512, ch_types='eeg')
            epochs = mne.EpochsArray(data=np.transpose(dd, (0, 2, 1)),
                                     info=info, events=None, tmin=0,
                                     baseline = (0, 0))
            epochs.set_montage(opt_ex['montage'])
            epochs.set_eeg_reference(ref_channels='average', projection=True)
            evoked = epochs.average()
            times = evoked.times
            # pdb.set_trace()
            # features['Rank'] = mne.compute_rank(epochs, tol=1e-6, tol_kind="relative")['eeg']
            
            # # # Divide data in two segments for when needed
            # # split_index = int(opt_ex['t_split'] * epochs.info['sfreq'])
            # # before_epochs, after_epochs = [], []
            # # for single_epoch in epochs.get_data():
            # #     before_epochs.append(single_epoch[:, :split_index])
            # #     after_epochs.append(single_epoch[:, split_index:])
    
            # # Spectral power
            # psd, freq = epochs.compute_psd().get_data(return_freqs = True)
            # for b, (start, stop) in opt_ex['eeg_bands'].items():
            #     indices = np.where((freq >= start) & (freq < stop))[0]
            #     tmp = []
            #     for ps in psd:
            #         tmp.append(np.mean(np.trapz(ps[:, indices], axis=-1)))
            #     features['PSD ' + b] = np.mean(tmp) 
    
            # # GFP
            # gfp = evoked.data.std(axis=0, ddof=0)
            # features['GFP'] = gfp
            # features['GFP peaks'] = np.max(gfp)
            # features['GFP timing'] = times[np.argmax(gfp)]                 
    
            # Fractal analysis
            # features['Fractal'] = compute_fractal_analysis(epochs, opt_ex)
    
            # PCI
            features['PCI'] = ParametrizePCI(epochs, opt_ex)
            
            # Source metrics
            # dipole_pos, stc_activity_area, leakage = compute_source_metrics(epochs, evoked, opt_ex)
            # features['STC Activity Area'] = stc_activity_area
            # features['Leakage'] = leakage
            # features['Dipole position'] = dipole_pos
                        
            if micro:
                output_file = os.path.join(opt_ex['res_dir'], f"features_{ix}_nns.pkl")
            else:
                output_file = os.path.join(opt_ex['res_dir'], f"features_{ix}_scs.pkl")
            with open(output_file, "wb") as file:
                pickle.dump(features, file)
            # pdb.set_trace()
            
    return 'Success'

def load_features(res_dir):
    features_micro = []
    features_ssep = []
    for ix in range(30):
        micro_file = os.path.join(res_dir, f"features_{ix}_nns.pkl")
        ssep_file = os.path.join(res_dir, f"features_{ix}_scs.pkl")

        if os.path.exists(micro_file):
            with open(micro_file, "rb") as file:
                features_micro.append(pickle.load(file))

        if os.path.exists(ssep_file):
            with open(ssep_file, "rb") as file:
                features_ssep.append(pickle.load(file))

    paired_features = list(zip(features_micro, features_ssep))
    
    return paired_features


def find_best_threshold(data, labels):
    # Ensure the data and labels are numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Sort data and labels by the data values
    sorted_indices = np.argsort(data)
    sorted_data = data[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    # Initialize variables to track the best threshold and maximum variance
    best_threshold = None
    max_accuracy = 0
    
    # Iterate through all possible thresholds
    for i in range(1, len(sorted_data)):
        threshold = (sorted_data[i - 1] + sorted_data[i]) / 2
        
        # Split data into two groups based on the threshold
        predictions = sorted_data > threshold
        
        # Calculate the classification accuracy
        accuracy = np.mean(predictions == sorted_labels)
        
        # Update the best threshold if the current accuracy is greater
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, max_accuracy

def calc_PCIst(signal_evk, times, full_return=False, **par):
    ''' Calculates PCIst (Perturbational Complexity Index based on State transitions) of a signal.
    Parameters
    ----------
    signal_evk : ndarray
        2D array (ch, times) containing signal.
    times : ndarray
        1D array (time,) containing timepoints (negative values are baseline).
    full_return : bool
        Returns multiple variables involved in PCI computation.
    **pars : dictionary
        Dictionary containing parameters (see dimensionality_reduction(),
        state_transition_quantification()and preprocess_signal() documentation).
        Example:
        >> par = {'baseline_window':(-400,-50), 'response_window':(0,300), 'k':1.2, 'min_snr':1.1,
        'max_var':99, 'embed':False,'n_steps':100}
        >> PCIst, PCIst_bydim = calc_PCIst(signal_evoked, times, **par)

    Returns
    -------
    float
        PCIst value
    OR (if full_return==True)
    dict
        Dictionary containing all variables from calculation including array 'dNSTn' with PCIst decomposition.
    '''
    if np.any(np.isnan(signal_evk)):
        print('Data contains nan values.')
        return 0

    # signal_evk, times = preprocess_signal(signal_evk, times, (par['baseline_window'][0],
    #                                                           par['response_window'][1]), **par)
    signal_evk, times = preprocess_signal(signal_evk, times, (par['response_window'][0],
                                                              par['baseline_window'][1]), **par)
    
    signal_svd, var_exp, eigenvalues, snrs = dimensionality_reduction(signal_evk, times, **par)
    STQ = state_transition_quantification(signal_svd, times, **par)

    PCI = np.sum(STQ['dNST'])

    if full_return:
        return {'PCI':PCI, **STQ, 'signal_evk':signal_evk, 'times':times, 'signal_svd':signal_svd,
                'eigenvalues':eigenvalues, 'var_exp':var_exp, 'snrs':snrs}
    return PCI

## DIMENSIONALITY REDUCTION
def dimensionality_reduction(signal, times, response_window, max_var=99, min_snr=1.1,
                             n_components=None, **kwargs):
    '''Returns principal components of signal according to SVD of the response.

    Calculates SVD at a given time interval (response_window) and uses the new basis to transform
    the whole signal yielding `n_components` principal components. The principal components are
    then selected to account for at least `max_var`% of the variance basesent in the signal's
    response.

    Parameters
    ----------
    signal : ndarray
        2D array (ch,time) containing signal.
    times : ndarray
        1D array (time,) containing timepoints
    response_window : tuple
        Signal's response time interval (ini,end).
    max_var: 0 < float <= 100
        Percentage of variance accounted for by the selected principal components.
    min_snr : float, optional
        Selects principal components with a signal-to-noise ratio (SNR) > min_snr.
    n_components : int, optional
        Number of principal components calculated (before selection).


    Returns
    -------
    np.ndarray
        2D array (ch,time) with selected principal components.
    np.ndarray
        1D array (n_components,) with `n_components` SVD eigenvalues of the signal's response.
    '''

    if not n_components:
        n_components = signal.shape[0]

    Vk, eigenvalues = get_svd(signal, times, response_window, n_components)
    var_exp = 100 * eigenvalues**2/np.sum(eigenvalues**2)

    signal_svd = apply_svd(signal, Vk)

    max_dim = calc_maxdim(eigenvalues, max_var)

    signal_svd = signal_svd[:max_dim, :]

    # if min_snr:
        # base_ini_ix = get_time_index(times, kwargs['baseline_window'][0])
        # base_end_ix = get_time_index(times, kwargs['baseline_window'][1])
        # resp_ini_ix = get_time_index(times, response_window[0])
        # resp_end_ix = get_time_index(times, response_window[1])
        # n_dims = np.size(signal_svd, 0)
        # snrs = np.zeros(n_dims)
        # for c in range(n_dims):
        #     resp_power = np.mean(np.square(signal_svd[c, resp_ini_ix:resp_end_ix]))
        #     base_power = np.mean(np.square(signal_svd[c, base_ini_ix:base_end_ix]))
        #     snrs[c] = np.sqrt(np.divide(resp_power, base_power))
    snrs = calc_snr(signal_svd, times, kwargs['baseline_window'], response_window)
    signal_svd = signal_svd[snrs > min_snr, :]
    snrs = snrs[snrs > min_snr]

    Nc = signal_svd.shape[0]

    return signal_svd, var_exp[:Nc], eigenvalues, snrs

def calc_snr(signal_svd, times, baseline_window, response_window):

    base_ini_ix = get_time_index(times, baseline_window[0])
    base_end_ix = get_time_index(times, baseline_window[1])
    resp_ini_ix = get_time_index(times, response_window[0])
    resp_end_ix = get_time_index(times, response_window[1])

    resp_power = np.mean(np.square(signal_svd[:,resp_ini_ix:resp_end_ix]), axis=1)
    base_power = np.mean(np.square(signal_svd[:,base_ini_ix:base_end_ix]), axis=1)
    snrs = np.sqrt(resp_power / base_power)
    return snrs

def get_svd(signal_evk, times, response_window, n_components):
    ini_t, end_t = response_window
    ini_ix = get_time_index(times, onset=ini_t)
    end_ix = get_time_index(times, onset=end_t)
    signal_resp = signal_evk[:, ini_ix:end_ix].T
    U, S, V = linalg.svd(signal_resp, full_matrices=False)
    V = V.T
    Vk = V[:, :n_components]
    eigenvalues = S[:n_components]
    return Vk, eigenvalues

def apply_svd(signal, V):
    '''Transforms signal according to SVD basis.'''
    return signal.T.dot(V).T

## STATE TRANSITION QUANTIFICATION
def state_transition_quantification(signal, times, k, baseline_window, response_window, embed=False,
                                    L=None, tau=None, n_steps=100, max_thr_p=1.0, **kwargs):
    ''' Receives selected principal components of perturbational signal and
    performs state transition quantification.

    Parameters
    ----------
    signal : ndarray
        2D array (component,time) containing signal (typically, the selected
        principal components).
    times : ndarray
        1D array (time,) containing timepoints
    k : float > 1
        Noise control parameter.
    baseline_window : tuple
        Signal's baseline time interval (ini,end).
    response_window : tuple
        Signal's response time interval (ini,end).
    embed : bool, optional
        Perform time-delay embedding.
    L : int
        Number of embedding dimensions.
    tau : int
        Number of timesamples of embedding delay
    n_steps : int, optional
        Number of steps used to search for the threshold that maximizes ∆NST.
        Search is performed by partitioning  the interval (defined from the median
        of the baseline’s distance matrix to the maximum of the response’s
        distance matrix) into ‘n_steps’ equal lengths.

    Returns
    -------
    float
        PCIst value.
    ndarray
        List containing component wise PCIst value (∆NSTn).
    '''

    n_dims = signal.shape[0]
    if n_dims == 0:
        print('No components --> PCIst=0')
        return {'dNST':np.array([]), 'n_dims':0}

    # EMBEDDING
    if embed:
        cut = (L-1)*tau
        times = times[cut:]
        temp_signal = np.zeros((n_dims, L, len(times)))
        for i in range(n_dims):
            temp_signal[i, :, :] = dimension_embedding(signal[i, :], L, tau)
        signal = temp_signal

    else:
        signal = signal[:, np.newaxis, :]

    # BASELINE AND RESPONSE DEFINITION
    base_ini_ix = get_time_index(times, baseline_window[0])
    base_end_ix = get_time_index(times, baseline_window[1])
    resp_ini_ix = get_time_index(times, response_window[0])
    resp_end_ix = get_time_index(times, response_window[1])
    n_baseline = len(times[base_ini_ix:base_end_ix])
    n_response = len(times[resp_ini_ix:resp_end_ix])

    if n_response <= 1 or n_baseline <= 1:
        print('Warning: Bad time interval defined.')

    baseline = signal[:, :, base_ini_ix:base_end_ix]
    response = signal[:, :, resp_ini_ix:resp_end_ix]

    # NST CALCULATION
        # Distance matrix
    D_base = np.zeros((n_dims, n_baseline, n_baseline))
    D_resp = np.zeros((n_dims, n_response, n_response))
        # Transition matrix
    T_base = np.zeros((n_steps, n_dims, n_baseline, n_baseline))
    T_resp = np.zeros((n_steps, n_dims, n_response, n_response))
        # Number of state transitions
    NST_base = np.zeros((n_steps, n_dims))
    NST_resp = np.zeros((n_steps, n_dims))
    thresholds = np.zeros((n_steps, n_dims))
    for i in range(n_dims):
        D_base[i, :, :] = recurrence_matrix(baseline[i, :, :], thr=None, mode='distance')
        D_resp[i, :, :] = recurrence_matrix(response[i, :, :], thr=None, mode='distance')
        min_thr = np.median(D_base[i, :, :].flatten())
        max_thr = np.max(D_resp[i, :, :].flatten()) * max_thr_p
        thresholds[:, i] = np.linspace(min_thr, max_thr, n_steps)
    for i in range(n_steps):
        for j in range(n_dims):
            T_base[i, j, :, :] = distance2transition(D_base[j, :, :], thresholds[i, j])
            T_resp[i, j, :, :] = distance2transition(D_resp[j, :, :], thresholds[i, j])

            NST_base[i, j] = np.sum(T_base[i, j, :, :])/n_baseline**2
            NST_resp[i, j] = np.sum(T_resp[i, j, :, :])/n_response**2

    # PCIST
    NST_diff = NST_resp - k * NST_base
    ixs = np.argmax(NST_diff, axis=0)
    max_thresholds = np.array([thresholds[ix, i] for ix, i in zip(ixs, range(n_dims))])
    dNST = np.array([NST_diff[ix, i] for ix, i in zip(ixs, range(n_dims))]) * n_response
    dNST = [x if x>0 else 0 for x in dNST]

    temp = np.zeros((n_dims, n_response, n_response))
    temp2 = np.zeros((n_dims, n_baseline, n_baseline))
    for i in range(n_dims):
        temp[i, :, :] = T_resp[ixs[i], i, :, :]
        temp2[i, :, :] = T_base[ixs[i], i, :, :]
    T_resp = temp
    T_base = temp2

    return {'dNST':dNST, 'n_dims':n_dims,
    'D_base':D_base, 'D_resp':D_resp, 'T_base':T_base,'T_resp':T_resp,
    'thresholds':thresholds, 'NST_diff':NST_diff, 'NST_resp':NST_resp, 'NST_base':NST_base,'max_thresholds':max_thresholds}


def recurrence_matrix(signal, mode, thr=None):
    ''' Calculates distance, recurrence or transition matrix. Signal can be
    embedded (m, n_times) or not (, n_times).

    Parameters
    ----------
    signal : ndarray
        Time-series; may be a 1D (time,) or a m-dimensional array (m, time) for
        time-delay embeddeding.
    mode : str
        Specifies calculated matrix: 'distance', 'recurrence' or 'transition'
    thr : float, optional
        If transition matrix is chosen (`mode`=='transition'), specifies threshold value.

    Returns
    -------
    ndarray
        2D array containing specified matrix.
    '''
    if len(signal.shape) == 1:
        signal = signal[np.newaxis, :]
    n_dims = signal.shape[0]
    n_times = signal.shape[1]

    R = np.zeros((n_dims, n_times, n_times))
    for i in range(n_dims):
        D = np.tile(signal[i, :], (n_times, 1))
        D = D - D.T
        R[i, :, :] = D
    R = np.linalg.norm(R, ord=2, axis=0)

    mask = (R <= thr) if thr else np.zeros(R.shape).astype(bool)
    if mode == 'distance':
        R[mask] = 0
        return R
    if mode == 'recurrence':
        return mask.astype(int)
    if mode == 'transition':
        return diff_matrix(mask.astype(int), symmetric=False)
    return 0

def distance2transition(dist_R, thr):
    ''' Receives 2D distance matrix and calculates transition matrix. '''
    mask = dist_R <= thr
    R = diff_matrix(mask.astype(int), symmetric=False)
    return R

def distance2recurrence(dist_R, thr):
    ''' Receives 2D distance matrix and calculates recurrence matrix. '''
    mask = dist_R <= thr
    return mask.astype(int)

def diff_matrix(A, symmetric=False):
    B = np.abs(np.diff(A))
    if B.shape[1] != B.shape[0]:
        B2 = np.zeros((B.shape[0], B.shape[1]+1))
        B2[:, :-1] = B
        B = B2
    if symmetric:
        B = (B + B.T)
        B[B > 0] = 1
    return B

def calc_maxdim(eigenvalues, max_var):
    ''' Get number of dimensions that accumulates at least `max_var`% of total variance'''
    if max_var == 100:
        return len(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1] # Sort in descending order
    var = eigenvalues ** 2
    var_p = 100 * var/np.sum(var)
    var_cum = np.cumsum(var_p)
    max_dim = len(eigenvalues) - np.sum(var_cum >= max_var) + 1
    return max_dim

def dimension_embedding(x, L, tau):
    '''
    Returns time-delay embedding of vector.
    Parameters
    ----------
    x : ndarray
        1D array time series.
    L : int
        Number of dimensions in the embedding.
    tau : int
        Number of samples in delay.
    Returns
    -------
    ndarray
        2D array containing embedded signal (L, time)

    '''
    assert len(x.shape) == 1, "x must be one-dimensional array (n_times,)"
    n_times = x.shape[0]
    s = np.zeros((L, n_times - (L-1) * tau))
    ini = (L-1) * tau if L > 1 else None
    s[0, :] = x[ini:]
    for i in range(1, L):
        ini = (L-i-1) * tau
        end = -i * tau
        s[i, :] = x[ini:end]
    return s

## PREPROCESS
def preprocess_signal(signal_evk, times, time_window, baseline_corr=False, resample=None,
                      avgref=False, **kwargs):
    assert signal_evk.shape[1] == len(times), 'Signal and Time arrays must be of the same size.'
    if avgref:
        signal_evk = avgreference(signal_evk)
    if baseline_corr:
        signal_evk = baseline_correct(signal_evk, times, delta=-50)
    t_ini, t_end = time_window
    ini_ix = get_time_index(times, t_ini)
    end_ix = get_time_index(times, t_end)
    signal_evk = signal_evk[:, ini_ix:end_ix]
    times = times[ini_ix:end_ix]
    if resample:
        signal_evk, times = undersample_signal(signal_evk, times, new_fs=resample)
    return signal_evk, times

def avgreference(signal):
    ''' Performs average reference to signal. '''
    new_signal = np.zeros(signal.shape)
    channels_mean = np.mean(signal, axis=0)[np.newaxis]
    new_signal = signal - channels_mean
    return new_signal

def undersample_signal(signal, times, new_fs):
    '''
    signal : (ch x times)
    times : (times,) [ms]
    new_fs : [hz]
    '''
    n_samples = int((times[-1]-times[0])/1000 * new_fs)
    new_signal_evk, new_times = scipy.signal.resample(signal, n_samples, t=times, axis=1)
    return new_signal_evk, new_times

def baseline_correct(Y, times, delta=0):
    ''' Baseline correct signal using times < delta '''
    newY = np.zeros(Y.shape)
    onset_ix = get_time_index(times, delta)
    baseline_mean = np.mean(Y[:, :onset_ix], axis=1)[np.newaxis]
    newY = Y - baseline_mean.T
    close_enough = np.all(np.isclose(np.mean(newY[:, :onset_ix], axis=1), 0, atol=1e-08))
    assert close_enough, "Baseline mean is not zero"
    return newY

def get_time_index(times, onset=0):
    ''' Returns index of first time greater then delta. For delta=0 gets index of
    first non-negative time.
    '''
    return np.sum(times < onset)