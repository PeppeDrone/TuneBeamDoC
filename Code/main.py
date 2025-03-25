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
from scipy.stats import mannwhitneyu, ttest_ind
from numpy.linalg import eig, inv
from matplotlib.patches import Ellipse
from skimage.measure import EllipseModel
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.cluster import KMeans
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.colors import ListedColormap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut, cross_val_score, LeaveOneGroupOut
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from statannot import add_stat_annotation
import pingouin as pg
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrow
import statsmodels.api as sm
# %% Execution parameters
plt.close('all')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams.update({'figure.max_open_warning': 0})  # It's simply annoying
mne.set_log_level("WARNING")
pd.options.mode.chained_assignment = None
np.set_printoptions(threshold=sys.maxsize)
pd.set_option('display.max_rows', None, 'display.max_columns', None)
plt.rcParams['keymap.quit'].append(' ')
code_dir = os.getcwd()
main_dir = code_dir.replace('\Code', '')
data_dir = main_dir + "\Data"
res_dir = main_dir + "\Results"
fig_dir = main_dir + "\Figures"

runfile(code_dir + '\support_tb.py',
        wdir=code_dir)


# %% Options definition
opt_ex = {'code_dir': code_dir,
          'fig_dir':fig_dir,
          'res_dir':res_dir,
         # --- EEG processing parameters
         'eeg_bands' : {'delta': (0.5, 4),
                      'theta': (4, 8),
                      'alpha': (8, 13),
                      'beta': (13, 30)},
         'montage': mne.channels.make_standard_montage("standard_1020"),
         'iterations_RANSAC': 10,
         'channels': ['AF7', 'AF3', 'Fp1', 'Fp2', 'AF4', 'AF8', 'F7', 'F5',
                      'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                      'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8', 'T3', 'C5',
                      'C3', 'C1', 'C2', 'C4', 'C6', 'T4', 'TP7', 'CP5',
                      'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8', 'T5', 'P5',
                      'P3', 'P1', 'P2', 'P4', 'P6', 'T6', 'Fpz', 'PO7',
                      'PO3', 'O1', 'O2', 'PO4', 'PO8', 'Oz', 'AFz', 'Fz',
                      'FCz', 'Cz', 'CPz', 'Pz', 'POz'],
         'roi_mapping' : {
             'Frontal': ['AF7', 'AF3', 'Fp1', 'Fp2', 'AF4', 'AF8'],
             'Fronto-Temporal': ['F7', 'F5', 'F3', 'F1', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'FT8'],
             'Temporal': ['T3', 'T4', 'T5', 'T6'],
             'Central': ['C5', 'C3', 'C1', 'C2', 'C4', 'C6', 'Cz'],
             'Centro-Parietal': ['TP7', 'CP5', 'CP3', 'CP1', 'CP2', 'CP4', 'CP6', 'TP8', 'CPz'],
             'Parietal': ['P5', 'P3', 'P1', 'P2', 'P4', 'P6', 'Pz'],
             'Occipital': ['O1', 'O2', 'Oz'],
             'Parieto-Occipital': ['PO7', 'PO3', 'PO4', 'PO8', 'POz'],
             'Midline': ['Fpz', 'AFz', 'Fz', 'FCz']
         },
         'fs': 512,
         't_split': 0.5,
         'pars_snr': [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
         'pars_k': [1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2],
         'side': [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1], # Stimulation side 0 is left, 1 is right
         'nns_thres': [7, 4, 4, 1, 4, 2, 4, 1, 4, 2, 1, 2, 1, 4, 1, 2, 3, 2, 1],
         'scs_thres': [19, 15, 9, 4, 8, 6, 9, 8, 15, 7, 4, 5, 13, 14,10, 10, 6, 14, 10],
         'diagnosis_level':[
               0, 0, 0, 
               0, 0, 0, 
               0, 2, 0,
               0, 1, 1,
               2, 0,
               1, 0, 0, 0, 1],
         'crs_adm':[3,7,4,
                    7,7,4,
                    6,13,5,
                    4,8,7,
                    13,6,
                    13,4,4,6,7],
         'crs_dim':[-1,7,5,
                    12,7,23,
                    5,-1,5,
                    23,23,11,
                    23,6,
                    15,10,6,12,7],
         'prognosis_level':[
              -1, 0, 0,
              2, 0, 3,
              0, -1, 0,
              3, 3, 2,
              3, 0,
              1, 1, 0, 2, 0],
         'covert_outcome':[
             0, 0, 0, 
             1, 0, 1, 
             0, 1, 0,
             1, 1, 1,
             1, 0,
             1, 1, 0, 1, 1],
         'consc_definition': {'-1':'Exitus',
                              '0':'UWS',
                              '1':'MCS-',
                              '2':'MCS+',
                              '3':'eMCS',
                              '4':'Healthy'},
         'color_diagnosis':[
               'red', 'red', 'red', 
               'red', 'red', 'red',
               'red', 'yellow', 'red',
               'red', 'orange', 'orange',
               'yellow', 'red',
               'orange', 'red', 'red', 'red', 'orange'],    
                  
          'color_improvement':[
              'black', 'gray', 'gray',
              'gray', 'gray', 'gray', 
              'gray',  'gray', 'gray',
              'gray', 'gray', 'gray',
              'gray','gray',
              'gray','gray','gray','gray','gray'],     
          
          'style_improvement':[
              '-', '-', '-',
              '--', '-', '--', 
              '-',  '-', '-',
              '--', '--', '--',
              '--','-',
              '-','--','-','--','-'
              ]   
         }

# %% Plot functions
def p_a_fig2(paired_features, opt_ex):

    amperes, amperes_sep = opt_ex['nns_thres'], opt_ex['scs_thres']
    status,statusp, color, colorp, style, crsadm, crsdim = opt_ex['diagnosis_level'],opt_ex['prognosis_level'], opt_ex['color_diagnosis'], opt_ex['color_improvement'], opt_ex['style_improvement'], opt_ex['crs_adm'], opt_ex['crs_dim']
    cc = np.argsort(crsadm) # cc = np.argsort(status)
    status_ord, color_ord, colorp_ord, style_ord, crsadmord, crsdimord = [status[i] for i in cc], [color[i] for i in cc], [colorp[i] for i in cc], [style[i] for i in cc], [crsadm[i] for i in cc], [crsdim[i] for i in cc]
    amperes_ord, amperes_sep_ord = [amperes[i] for i in cc], [amperes_sep[i] for i in cc]
    times = np.linspace(0, 1, 512)
    improvement = [sp-s for sp, s in zip(statusp, status)]
    improvement = np.ravel([1 if imp > 0 else 0 for imp in improvement])
    improvement_ord = [improvement[i] for i in cc]
    
    fig_2 = plt.figure(figsize=(22, 9))
    gs = gridspec.GridSpec(3, 7, figure=fig_2, hspace=0.8, wspace=0.4)
    axA = fig_2.add_subplot(gs[:2, :2])
    axB = fig_2.add_subplot(gs[:2, 2])
    axC = fig_2.add_subplot(gs[2, :2])
    axD = fig_2.add_subplot(gs[:2, 4:6])
    axE = fig_2.add_subplot(gs[:2, 6])
    axF = fig_2.add_subplot(gs[2, 4:6])
   
    gfpnns, gfpscs, gfpnnsp, gfpscsp, gfpnnst, gfpscst = [], [], [], [], [], []
    for idx, (features_nns, features_scs) in enumerate(paired_features):
        gfpnns.append(features_nns['GFP'])
        gfpscs.append(features_scs['GFP'])
        gfpnnsp.append(features_nns['GFP peaks'])
        gfpscsp.append(features_scs['GFP peaks'])
        gfpnnst.append(features_nns['GFP timing'])
        gfpscst.append(features_scs['GFP timing'])
    for idx in range(len(paired_features)):
        iddx = cc[idx]
        axA.plot(times, gfpnns[iddx]/amperes[iddx], color=color[iddx])
        axD.plot(times, gfpscs[iddx]/amperes_sep[iddx], color=color[iddx])

        # axB.scatter(x=iddx, y=gfpnnsp[iddx]/amperes[iddx], color=color[iddx])
        # axB.scatter(x=iddx, y=gfpnnsp[iddx]/amperes[iddx], facecolors='none', edgecolors=colorp[iddx], linestyle=style[iddx], s=130, linewidth=1.6)
        # axC.scatter(x=gfpnnst[iddx], y=iddx, color=color[iddx])
        # axC.scatter(y=iddx, x=gfpnnst[iddx], facecolors='none', edgecolors=colorp[iddx], linestyle=style[iddx], s=130, linewidth=1.6)
        # axE.scatter(x=iddx, y=gfpscsp[iddx]/amperes_sep[iddx], color=color[iddx])
        # axE.scatter(x=iddx, y=gfpscsp[iddx]/amperes_sep[iddx], facecolors='none', edgecolors=colorp[iddx], linestyle=style[iddx], s=130, linewidth=1.6)
        # axF.scatter(x=gfpscst[iddx], y=iddx, color=color[iddx])
        # axF.scatter(y=iddx, x=gfpscst[iddx], facecolors='none', edgecolors=colorp[iddx], linestyle=style[iddx], s=130, linewidth=1.6)
        
    axB.scatter(x=crsadmord, y=np.array(gfpnnsp)[cc]/amperes_ord,
                color=color_ord)
    axB.scatter(x=crsadmord, y=np.array(gfpnnsp)[cc]/amperes_ord,
                facecolors='none', edgecolors=colorp_ord, linestyle=style_ord, s=130, linewidth=1.6)
    peak = np.array(gfpnnsp)[cc]/amperes_ord
    for idx in range(len(paired_features)):
        if crsdimord[idx] >=0:
            arrow = FancyArrow(crsadmord[idx], peak[idx], crsdimord[idx] - crsadmord[idx], 0,  # (x, y, dx, dy)
                   width=0.1, 
                   head_width=0.3, 
                   head_length=0.6,
                   alpha = 0.4,
                   length_includes_head=True,
                   color='gray')#color_ord[idx])

            axB.add_patch(arrow)
    axC.scatter(x=np.array(gfpnnst)[cc], y=crsadmord,
                color=color_ord)
    axC.scatter(y=crsadmord, x=np.array(gfpnnst)[cc],
                facecolors='none', edgecolors=colorp_ord, linestyle=style_ord, s=130, linewidth=1.6)
    axE.scatter(x=crsadmord, y=np.array(gfpscsp)[cc]/amperes_sep_ord,
                color=color_ord)
    axE.scatter(x=crsadmord, y=np.array(gfpscsp)[cc]/amperes_sep_ord,
                facecolors='none', edgecolors=colorp_ord, linestyle=style_ord, s=130, linewidth=1.6)
    peak = np.array(gfpscsp)[cc]/amperes_sep_ord
    for idx in range(len(paired_features)):
        if crsdimord[idx] >=0:
            arrow = FancyArrow(crsadmord[idx], peak[idx], crsdimord[idx] - crsadmord[idx], 0,  # (x, y, dx, dy)
                   width=0.02, 
                   head_width=0.05, 
                   head_length=0.6,
                   alpha = 0.4,
                   length_includes_head=True,
                   color='gray')#color_ord[idx])
            axE.add_patch(arrow)
    axF.scatter(x=np.array(gfpscst)[cc], y=crsadmord,
                color=color_ord)
    axF.scatter(y=crsadmord, x=np.array(gfpscst)[cc],
                facecolors='none', edgecolors=colorp_ord, linestyle=style_ord, s=130, linewidth=1.6)
    # pdb.set_trace()
    axA.set_ylabel(r'Normalized GFP [$\mu$V/$mA$]')
    axA.set_xlabel(r'Time [s]')
    axA.set_ylim([0, 7])
    axB.set_ylim([0, 7])
    axB.set_xlim([1, 23.3])
    # axB.set_xticks([])
    axB.set_xlabel('CRS-R total score\nat $T_0$')
    axA.set_xlim([-0.05, 1.05])
    axC.set_xlim([-0.05, 1.05])
    axC.set_ylim([1, 15])
    axC.set_ylabel('CRS-R total score\nat $T_0$')
    axB.set_ylabel(r'Absolute peak [$\mu$V/$mA$]')
    axC.set_xlabel(r'Peak latency [s]')
    
    axD.set_ylabel(r'Normalized GFP [$\mu$V/$mA$]')
    axD.set_xlabel(r'Time [s]')
    axD.set_ylim([0, 1.5])
    axE.set_ylim([0, 1.5])
    axE.set_xlim([1, 23.3])
    axE.set_xlabel('CRS-R total score\nat $T_0$')
    axD.set_xlim([-0.05, 1.05])
    axF.set_ylabel('CRS-R total score\nat $T_0$')
    axF.set_xlim([-0.05, 1.05])
    axF.set_ylim([1, 15])
    axE.set_ylabel(r'Absolute peak [$\mu$V/$mA$]')
    axF.set_xlabel(r'Peak latency [s]')
    
    # Add legends
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='w', markersize=10, label='Consciousness state at $T_0$'),
        plt.Line2D([0], [0], marker='o', color='red', markerfacecolor='red', markersize=10, label='UWS'),
        plt.Line2D([0], [0], marker='o', color='orange', markerfacecolor='orange', markersize=10, label='MCS-'),
        plt.Line2D([0], [0], marker='o', color='yellow', markerfacecolor='yellow', markersize=10, label='MCS+'),
        
        plt.scatter([0], [0], marker='o', color='w', facecolor='w', s=80, label='Consciousness state at $T_1$'),
        plt.scatter([0], [0], marker='o', color='black', facecolor='w', s=80, label='Exitus for clinical complexity'),
        plt.scatter([0], [0], marker='o', color='gray', facecolor='w', s=80, label='Not improved'),
        plt.scatter([0], [0], marker='o', linestyle= 'dashed', color='gray', facecolor='w', s=80, label='Improved')
    ]
    axA.legend(handles=legend_elements, loc='upper right')
    axD.legend(handles=legend_elements, loc='upper right')
    fig_2.savefig(os.path.join(opt_ex['fig_dir'], "Figure_2.svg"), format = 'svg')
    # pdb.set_trace()
    
    # Compute median and IQR for peaks and latencies divided by improvement groups
    peaks_nns = np.array(gfpnnsp)[cc] / np.array(amperes_ord)
    peaks_scs = np.array(gfpscsp)[cc] / np.array(amperes_sep_ord)
    latencies_nns = np.array(gfpnnst)[cc]
    latencies_scs = np.array(gfpscst)[cc]
    
    improvement_groups = np.array(improvement_ord)
    
    def compute_stats(data, groups):
        group_0 = data[groups == 0]
        group_1 = data[groups == 1]
        median_0, iqr_0 = np.median(group_0), np.percentile(group_0, 75) - np.percentile(group_0, 25)
        median_1, iqr_1 = np.median(group_1), np.percentile(group_1, 75) - np.percentile(group_1, 25)
        u_stat, p_value = mannwhitneyu(group_0, group_1)
        return (median_0, iqr_0), (median_1, iqr_1), u_stat, p_value
    
    peaks_nns_stats = compute_stats(peaks_nns, improvement_groups)
    peaks_scs_stats = compute_stats(peaks_scs, improvement_groups)
    latencies_nns_stats = compute_stats(latencies_nns, improvement_groups)
    latencies_scs_stats = compute_stats(latencies_scs, improvement_groups)
    
    print("NNS Peaks - Group 0 (Not Improved): Median =", peaks_nns_stats[0][0], "IQR =", peaks_nns_stats[0][1])
    print("NNS Peaks - Group 1 (Improved): Median =", peaks_nns_stats[1][0], "IQR =", peaks_nns_stats[1][1])
    print("Mann-Whitney U Test: U =", peaks_nns_stats[2], "p =", peaks_nns_stats[3])
    
    print("SCS Peaks - Group 0 (Not Improved): Median =", peaks_scs_stats[0][0], "IQR =", peaks_scs_stats[0][1])
    print("SCS Peaks - Group 1 (Improved): Median =", peaks_scs_stats[1][0], "IQR =", peaks_scs_stats[1][1])
    print("Mann-Whitney U Test: U =", peaks_scs_stats[2], "p =", peaks_scs_stats[3])
    
    print("NNS Latencies - Group 0 (Not Improved): Median =", latencies_nns_stats[0][0], "IQR =", latencies_nns_stats[0][1])
    print("NNS Latencies - Group 1 (Improved): Median =", latencies_nns_stats[1][0], "IQR =", latencies_nns_stats[1][1])
    print("Mann-Whitney U Test: U =", latencies_nns_stats[2], "p =", latencies_nns_stats[3])
    
    print("SCS Latencies - Group 0 (Not Improved): Median =", latencies_scs_stats[0][0], "IQR =", latencies_scs_stats[0][1])
    print("SCS Latencies - Group 1 (Improved): Median =", latencies_scs_stats[1][0], "IQR =", latencies_scs_stats[1][1])
    print("Mann-Whitney U Test: U =", latencies_scs_stats[2], "p =", latencies_scs_stats[3])


    return 'Success'

def p_a_fig3(paired_features, opt_ex):
    amperes, amperes_sep = opt_ex['nns_thres'], opt_ex['scs_thres']
    status,statusp, color, colorp, style, crsadm, crsdim = opt_ex['diagnosis_level'],opt_ex['prognosis_level'], opt_ex['color_diagnosis'], opt_ex['color_improvement'], opt_ex['style_improvement'], opt_ex['crs_adm'], opt_ex['crs_dim']
    cc = np.argsort(crsadm) # cc = np.argsort(status)
    status_ord, color_ord, colorp_ord, style_ord, crsadm_ord, crsdim_ord = [status[i] for i in cc], [color[i] for i in cc], [colorp[i] for i in cc], [style[i] for i in cc], [crsadm[i] for i in cc], [crsdim[i] for i in cc]
    times = np.linspace(0, 1, 512)
    improvement = [sp-s for sp, s in zip(statusp, status)]
    improvement = np.ravel([1 if imp > 0 else 0 for imp in improvement])
    improvement_ord = [improvement[i] for i in cc]
    
    df = {'pci_nns': [], 'pci_scs': [], 'k': [], 'min_snr': []}
    for idx, (features_nns, features_scs) in enumerate(paired_features):
        df['pci_nns'].append(features_nns['PCI']['pci'])
        df['pci_scs'].append(features_scs['PCI']['pci'])
        df['k'].append(features_nns['PCI']['k'])
        df['min_snr'].append(features_nns['PCI']['min_snr'])
    
    df['pci_nns'] = np.hstack(df['pci_nns'])
    df['pci_scs'] = np.hstack(df['pci_scs'])
    df['k'] = np.hstack(df['k'])
    df['min_snr'] = np.hstack(df['min_snr'])
    c1, c2 = np.array(df['k']) == 1.2, np.array(df['min_snr']) == 1.1 # k = 1.2
    where_paper = np.squeeze(np.argwhere(c1 & c2))  
    
    figure_3_pci, ax_pci = plt.subplots(1, 2, figsize=(8,4))
    
    ll = np.array(df['pci_nns'][where_paper]) / np.array(opt_ex['nns_thres'])
    print(mannwhitneyu(ll[improvement==0], ll[improvement==1]))
    ax_pci[0].scatter(crsadm_ord, ll[cc], c=color_ord, marker='o', s=70)
    ax_pci[0].scatter(crsadm_ord, ll[cc], facecolors='none', 
                    edgecolors=colorp_ord, linestyle=style_ord, s=130, linewidth=1.6)
    for idx in range(len(paired_features)):
        if crsdim[idx] >=0:
            arrow = FancyArrow(crsadm[idx], ll[idx], crsdim[idx] - crsadm[idx], 0,  # (x, y, dx, dy)
                   width=0.1, 
                   head_width=0.3, 
                   head_length=0.6,
                   alpha = 0.4,
                   length_includes_head=True,
                   color='gray')#color_ord[idx])

            ax_pci[0].add_patch(arrow)

    ll = np.array(df['pci_scs'][where_paper]) / np.array(opt_ex['scs_thres'])
    print(mannwhitneyu(ll[improvement==0], ll[improvement==1]))
    ax_pci[1].scatter(crsadm_ord, ll[cc], c=color_ord, marker='o', s=70)
    ax_pci[1].scatter(crsadm_ord, ll[cc], facecolors='none', 
                    edgecolors=colorp_ord, linestyle=style_ord, s=130, linewidth=1.6)
    for idx in range(len(paired_features)):
        if crsdim[idx] >=0:
            arrow = FancyArrow(crsadm[idx], ll[idx], crsdim[idx] - crsadm[idx], 0,  # (x, y, dx, dy)
                   width=0.024, 
                   head_width=0.08, 
                   head_length=0.6,
                   alpha = 0.4,
                   length_includes_head=True,
                   color='gray')#color_ord[idx])

            ax_pci[1].add_patch(arrow)
    
    
    ax_pci[0].set_ylabel('Normalized PCI$_{st}$ [#/mA]')
    ax_pci[1].set_ylabel('Normalized PCI$_{st}$ [#/mA]')
    ax_pci[0].set_title('NNS', fontsize = 12)
    ax_pci[1].set_title('SCS', fontsize = 12)
    ax_pci[0].set_xlabel('CRS-R total score at $T_0$')
    ax_pci[1].set_xlabel('CRS-R total score at $T_0$')
    # ax_pci[0].set_xticks([])
    ax_pci[0].spines['top'].set_visible(False)
    ax_pci[0].spines['bottom'].set_linewidth(1.3)
    ax_pci[0].spines['left'].set_linewidth(1.3)
    ax_pci[0].spines['right'].set_visible(False)
    # ax_pci[1].set_xticks([])
    ax_pci[1].spines['top'].set_visible(False)
    ax_pci[1].spines['bottom'].set_linewidth(1.3)
    ax_pci[1].spines['left'].set_linewidth(1.3)
    ax_pci[1].spines['right'].set_visible(False)
    
    figure_3_pci.tight_layout()
    figure_3_pci.savefig(os.path.join(opt_ex['fig_dir'], 'Figure_3_PerturbComplexityIndex.svg'), format='svg')
    # pdb.set_trace()


    fract_nns, fract_scs = [], []
    for idx, (features_nns, features_scs) in enumerate(paired_features):
        try:
            fract_nns.append(features_nns['Fractal'])
        except:
            fract_nns.append(features_nns['Fractal before'])
        try:
            fract_scs.append(features_scs['Fractal'])
        except:
            fract_scs.append(features_scs['Fractal before'])
    fract_nns = pd.concat(fract_nns, axis=1).T
    fract_scs = pd.concat(fract_scs, axis=1).T
    fig_ABCD_upper, axes = plt.subplots(2, 5, figsize=(10,6))
    fig_ABCD_upper.subplots_adjust(hspace=0.3, wspace=0.2)
    keys_base = ['Hjorth_', 'AttEn']
    keys_nice = ['Hjorth complexity', 'Attention entropy']
    roi_mapping = opt_ex['roi_mapping']
    Vlimsnns = [[2,5], [1, 2.5]]
    Vlimsscs = [[2,5], [1, 2.5]]
    for ix, (kb, kn, vlnns,vlscs) in enumerate(zip(keys_base, keys_nice, Vlimsnns, Vlimsscs)):
        ss_nns = fract_nns.filter(regex=kb).values
        ss_scs = fract_scs.filter(regex=kb).values
        # Compute average values across ROI mappings
        roi_averages = {'NNS': {}, 'SCS': {}}
        for roi, channels in roi_mapping.items():
            channel_indices = [opt_ex['channels'].index(ch) for ch in channels if ch in opt_ex['channels']]
            roi_averages['NNS'][roi] = np.mean(ss_nns[:, channel_indices], axis=1)
            roi_averages['SCS'][roi] = np.mean(ss_scs[:, channel_indices], axis=1)
        roi_averages['NNS']['Status'] = improvement
        roi_averages['SCS']['Status'] = improvement

        pd.DataFrame.from_dict(roi_averages['NNS']).to_excel('ROI_fractal_nns_'+kb+'.xlsx')
        pd.DataFrame.from_dict(roi_averages['SCS']).to_excel('ROI_fractal_scs_'+kb+'.xlsx')
        
        info = mne.create_info(ch_names=opt_ex['channels'], sfreq=512, ch_types='eeg')
        info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    
        v_minnns, v_maxnns = vlnns[0], vlnns[1]
        v_minscs, v_maxscs = vlscs[0], vlscs[1]

        im1_nns, cm1_nns = mne.viz.plot_topomap(np.mean(ss_nns[improvement == 0], axis=0), vlim=(v_minnns, v_maxnns),
                                                pos=info, axes=axes[ix, 0], show=False)
        im2_nns, cm2_nns = mne.viz.plot_topomap(np.mean(ss_nns[improvement == 1], axis=0), vlim=(v_minnns, v_maxnns),
                                                pos=info, axes=axes[ix, 1], show=False)
    
        im1_scs, cm1_scs = mne.viz.plot_topomap(np.mean(ss_scs[improvement == 0], axis=0), vlim=(v_minscs, v_maxscs),
                                                pos=info, axes=axes[ix, 2], show=False)
        im2_scs, cm2_scs = mne.viz.plot_topomap(np.mean(ss_scs[improvement == 1], axis=0), vlim=(v_minscs, v_maxscs),
                                                pos=info, axes=axes[ix, 3], show=False)
        axes[ix, 0].set_title('Not improved')  
        axes[ix, 1].set_title('Improved')
        axes[ix, 0].set_ylabel(kn, fontsize = 14)
        axes[ix, 2].set_title('Not improved')  
        axes[ix, 3].set_title('Improved')
        axes[ix,4].set_frame_on(False)  # Removes the box around the plot
        axes[ix,4].set_xticks([])
        axes[ix,4].set_yticks([])
        cbar = fig_ABCD_upper.colorbar(im2_nns, ax=axes[ix, 4], shrink=0.8)
        cbar.ax.tick_params(labelsize = 12)

    plt.tight_layout()
    plt.show()
    fig_ABCD_upper.savefig(os.path.join(opt_ex['fig_dir'], "Figure_3_RestingComplexity.svg"), format = 'svg')        


    # GFP attractors!
    attractors_nns, attractors_scs = [], []
    sd1_over_sd2 = {'SD1':[],
                    'SD2':[],
                    'Type':[],
                    'ID':[],
                    'Status':[],
                    'Statuss':[]}
    theta = -np.pi / 4
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    for idx, (features_nns, features_scs) in enumerate(paired_features):
        attractors_nns.append(nk.complexity_embedding(features_nns['GFP'])[:,:2] @ rotation_matrix)
        attractors_scs.append(nk.complexity_embedding(features_scs['GFP'])[:,:2] @ rotation_matrix)
    
    fig_3_F, axs = plt.subplots(5,4,figsize = (8,6))
    axs = axs.ravel()
    dict_imp = {'0':'Not improved',
                '1':'Improved'}
    for ix, ax in enumerate(axs):
        if ix <= 18:
            ixx = cc[ix]        
        
            ax.plot(attractors_scs[ixx][:, 0], attractors_scs[ixx][:, 1], label = 'SCS', color = 'grey', clip_on = True)
            ell = EllipseModel()
            ell.estimate(np.transpose(np.squeeze(np.stack([attractors_scs[ixx][:, 0], attractors_scs[ixx][:, 1]]))))
            xc, yc, a, b, theta = ell.params
            sd1_over_sd2['SD1'].append(a)
            sd1_over_sd2['SD2'].append(b)
            sd1_over_sd2['Type'].append(0)
            sd1_over_sd2['ID'].append(ixx)
            sd1_over_sd2['Status'].append(dict_imp[str(improvement[ixx])])
            sd1_over_sd2['Statuss'].append(improvement[ixx])
    
            ax.plot(attractors_nns[ixx][:, 0], attractors_nns[ixx][:, 1], label = 'NNS', color = 'black', linestyle = '--', clip_on = True)
            ell = EllipseModel()
            ell.estimate(np.transpose(np.squeeze(np.stack([attractors_nns[ixx][:, 0], attractors_nns[ixx][:, 1]]))))
            xc, yc, a, b, theta = ell.params
            sd1_over_sd2['SD1'].append(a)
            sd1_over_sd2['SD2'].append(b)
            sd1_over_sd2['Type'].append(1)
            sd1_over_sd2['ID'].append(ixx)
            sd1_over_sd2['Status'].append(dict_imp[str(improvement[ixx])])
            sd1_over_sd2['Statuss'].append(improvement[ixx])
    
            # ax.set_aspect('equal')        
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            # ax.spines['left'].set_visible(False)
            # ax.spines['bottom'].set_visible(False)
            ax.set_xlim([-0.5,0.5])
            ax.set_ylim([0,8])
            # ax.set_xticks([])
            # ax.set_yticks([])
            
            ax.scatter(x=0.4, y=6,
                        color=color_ord[ix], s = 140)
            ax.scatter(x=0.4, y=6,
                        facecolors='none', edgecolors=colorp_ord[ix], linestyle=style_ord[ix], s=180, linewidth=1.6)
    
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])
    axs[len(cc)-1].legend(fontsize = 12, title = 'Stimulation type')
    # axs[-1].set_frame_on(False)
    # axs[-1].set_xticks([])
    # axs[-1].set_yticks([])
    axs[-1].set_xlim([-0.5,0.5])
    axs[-1].set_ylim([0,8])
    axs[-1].spines['left'].set_linewidth(1.3)
    axs[-1].spines['bottom'].set_linewidth(1.3)
    axs[-1].spines['top'].set_visible(False)
    axs[-1].spines['right'].set_visible(False)

    fig_3_F.tight_layout()
    fig_3_F.savefig(os.path.join(opt_ex['fig_dir'], 'Figure_3_GFPComplexity.svg'), format='svg')
    
    fig, axs = plt.subplots(1, 2, figsize=(6.5,3))
    sns.boxplot(data = sd1_over_sd2, y = 'SD1', x = 'Type', hue = 'Statuss', ax=axs[0], palette='Set2')
    add_stat_annotation(axs[0], data=sd1_over_sd2, y = 'SD1', x='Type', hue='Statuss',
                        box_pairs=[((i, 0), (i, 1)) for i in range(2)],
                        test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    axs[0].set_xticklabels(['NNS','SCS'])
    axs[0].set_ylabel('SD1', fontsize = 12)
    
    sns.boxplot(data = sd1_over_sd2, y = 'SD2', x = 'Type', hue = 'Statuss', ax=axs[1], palette='Set2')
    add_stat_annotation(axs[1], data=sd1_over_sd2, y = 'SD2', x='Type', hue='Statuss',
                        box_pairs=[((i, 0), (i, 1)) for i in range(2)],
                        test='Mann-Whitney')
    axs[1].set_ylabel('SD2', fontsize = 12)
    axs[1].set_xticklabels(['NNS','SCS'])
    axs[1].legend(labels = ['Not improved','Improved'])
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(opt_ex['fig_dir'], 'Figure_3_SD1_SD2_group.svg'), format='svg')
    pd.DataFrame.from_dict(sd1_over_sd2).to_excel('SD12.xlsx')
    # pdb.set_trace()

    return sd1_over_sd2

def p_a_fig4(paired_features, opt_ex):
    amperes, amperes_sep = opt_ex['nns_thres'], opt_ex['scs_thres']   
    status, statusp, color, colorp, style = opt_ex['diagnosis_level'], opt_ex['prognosis_level'], opt_ex['color_diagnosis'], opt_ex['color_improvement'], opt_ex['style_improvement']
    improvement = [sp-s for sp, s in zip(statusp, status)]
    improvement = np.ravel([1 if imp > 0 else 0 for imp in improvement])

    covert = opt_ex['covert_outcome']
    # pdb.set_trace()
    # improvement.pop(7)
    # improvement.pop(0)
    # color.pop(7)
    # color.pop(0)
    # colorp.pop(7)
    # colorp.pop(0)
    # style.pop(7)
    # style.pop(0)
    
    
    df_inf_nns = {'PCI':[],
               # 'PSD d':[], 'PSD t':[],
               # 'PSD a':[], 
               # 'PSD b':[],
              'GFP':[], 
               # 'SD1': [], 'SD2':[]
              }
    df_inf_scs = {'PCI':[],
               # 'PSD d':[], 'PSD t':[],
               # 'PSD a':[],
               # 'PSD b':[],
              'GFP':[],
               # 'SD1': [], 'SD2':[]
              }

    sd12 = opt_ex['sd1_over_sd2']
    for idx, (features_nns, features_scs) in enumerate(paired_features):
        c1 = np.array(paired_features[0][0]['PCI']['k']) == 1.2
        c2 = np.array(paired_features[0][0]['PCI']['min_snr']) == 1.1 # k = 1.2
        where_paper = np.squeeze(np.argwhere(c1 & c2)) 
        df_inf_nns['PCI'].append(features_nns['PCI']['pci'][where_paper]/amperes[idx])
        # df_inf_nns['PSD d'].append(features_nns['PSD delta'])
        # df_inf_nns['PSD t'].append(features_nns['PSD theta'])
        # df_inf_nns['PSD a'].append(features_nns['PSD alpha'])
        # df_inf_nns['PSD b'].append(features_nns['PSD beta'])
        df_inf_nns['GFP'].append(features_nns['GFP peaks']/amperes[idx])
        # df_inf_nns['SD1'].append(sd12['SD1'][np.argwhere((np.array(sd12['ID']) == idx) & (np.array(sd12['Type']) == 0))[0][0]])
        # df_inf_nns['SD2'].append(sd12['SD2'][np.argwhere((np.array(sd12['ID']) == idx) & (np.array(sd12['Type']) == 0))[0][0]])
        
        df_inf_scs['PCI'].append(features_scs['PCI']['pci'][where_paper]/amperes_sep[idx])
        # df_inf_scs['PSD d'].append(features_scs['PSD delta'])
        # df_inf_scs['PSD t'].append(features_scs['PSD theta'])
        # df_inf_scs['PSD a'].append(features_scs['PSD alpha'])
        # df_inf_scs['PSD b'].append(features_scs['PSD beta'])
        df_inf_scs['GFP'].append(features_scs['GFP peaks']/amperes_sep[idx])
        # df_inf_scs['SD1'].append(sd12['SD1'][np.argwhere((np.array(sd12['ID']) == idx) & (np.array(sd12['Type']) == 1))[0][0]])
        # df_inf_scs['SD2'].append(sd12['SD2'][np.argwhere((np.array(sd12['ID']) == idx) & (np.array(sd12['Type']) == 1))[0][0]])
        
    # for d in df_inf_nns:
    #     df_inf_nns[d].pop(7)
    #     df_inf_nns[d].pop(0)      
    # for d in df_inf_scs:
    #     df_inf_scs[d].pop(7)
    #     df_inf_scs[d].pop(0)   
    
    # keysplot = ['PCI', r'PSD$_{\delta}$',r'PSD$_{\theta}$',r'PSD$_{\alpha}$',r'PSD$_{\beta}$','GFP Peak','SD1','SD2']
    keysplot = ['PCI','GFP Peak']

    # ixpl = 0
    # fig, axs = plt.subplots(1,2, figsize = (7,4))
    # for r1, r2, lab in zip(list(df_inf_nns.values()), list(df_inf_scs.values()), keysplot):
    #     fpr, tpr, _ = roc_curve(y_roc, r1)
    #     roc_auc = auc(fpr, tpr)
    #     axs[0].plot(fpr, tpr, lw=2, label=f' {lab} (AuC = {roc_auc:.2f})')
        
    
    #     fpr, tpr, _ = roc_curve(y_roc, r2)
    #     roc_auc = auc(fpr, tpr)
    #     axs[1].plot(fpr, tpr, lw=2, label=f' {lab} (AuC = {roc_auc:.2f})')
            
    # axs[0].spines['right'].set_visible(False)
    # axs[0].spines['top'].set_visible(False)
    # axs[0].set_xlim([0.0, 1.0])
    # axs[0].set_ylim([0.0, 1.05])
    # axs[0].set_xlabel('False Positive Rate')
    # axs[0].set_ylabel('True Positive Rate')    
    # axs[1].set_xlim([0.0, 1.0])
    # axs[1].set_ylim([0.0, 1.05])
    # axs[1].set_xlabel('False Positive Rate')
    # axs[1].set_ylabel('True Positive Rate')
    # axs[1].spines['right'].set_visible(False)
    # axs[1].spines['top'].set_visible(False)  
    # axs[0].axis('square')
    # axs[1].axis('square')
    # axs[0].plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label = 'y = x')
    # axs[0].legend()
    # axs[1].plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label = 'y = x')
    # axs[1].legend()
    # for spine in axs[0].spines.values():
    #     spine.set_linewidth(1.4)
    # for spine in axs[1].spines.values():
    #     spine.set_linewidth(1.4)
    # axs[0].set_title('NNS')
    # axs[1].set_title('SCS')
    # fig.savefig(os.path.join(opt_ex['fig_dir'], 'Figure_4_Univariate.svg'), format='svg')
    
    # # Plot the odds ratios with error bars
    # fig2, axs = plt.subplots(2,1,figsize=(8, 6))
    
    # or_values = []
    # ci_lower = []
    # ci_upper = []
    # for predictor in df_inf_nns.keys():
    #     X = df_inf_nns[predictor]
    #     X = sm.add_constant(X)  # Add intercept
    #     model = sm.Logit(y_roc, X).fit(disp=1)
        
    #     # Extract the coefficient and its 95% confidence interval for the predictor
    #     coef = model.params[-1]
    #     conf_int = model.conf_int()[:,-1]
    #     or_values.append(np.exp(coef))
    #     ci_lower.append(np.exp(conf_int[0]))
    #     ci_upper.append(np.exp(conf_int[1])) 
    #     print(model.pvalues[1])
        
    # # Convert lists to NumPy arrays for plotting
    # or_values = np.array(or_values)
    # ci_lower = np.array(ci_lower)
    # ci_upper = np.array(ci_upper)
    # error_lower = np.abs(or_values - ci_lower)
    # error_upper = np.abs(ci_upper - or_values)  
    
    # axs[0].errorbar(range(len(df_inf_nns.keys())), or_values, yerr=[error_lower, error_upper],
    #             fmt='o', capsize=5, capthick=2, elinewidth=1.5)
    # axs[0].set_xticks(range(len(df_inf_nns.keys())))
    # # ax.set_xticklabels(display_names)
    # ax.set_xlabel("Predictor")
    # ax.set_ylabel("Odds Ratio")
    # ax.set_title("Univariate Logistic Regression: Odds Ratios with 95% CI")
    # ax.axhline(1, color='grey', lw=1, linestyle='--')  # Reference line at OR=1
    # ax.grid(True, linestyle='--', alpha=0.6)
    # plt.tight_layout()
    # plt.show()
        
    # pdb.set_trace()
    y_roc = np.array(covert)
    ##
    data1 = np.array(list(df_inf_nns.values())).T[:,[0,1]]
    print(np.mean(data1, axis = 0))
    print(np.std(data1, axis = 0))
    data2 = np.array(list(df_inf_scs.values())).T[:,[0,1]]
    scaler = StandardScaler()
    data1 = scaler.fit_transform(data1)
    scaler = StandardScaler()
    data2 = scaler.fit_transform(data2)
    fig, axs = plt.subplots(3, 2, figsize=(18, 6))
    h = .01 
    
    # XY plane
    axs[0,0].scatter(data1[:, 0], data1[:, 1], c=color, s=50, cmap='viridis')
    axs[0,0].scatter(data1[:, 0], data1[:, 1],
                 facecolors='none', edgecolors=colorp, linestyle=style, s=130, linewidth=1.6)
    axs[0,0].set_xlabel('Peak [$\mu$V/mA]')
    axs[0,0].set_ylabel('PCI [#/mA]')
    axs[0,0].axis('square')

    # # XZ plane
    axs[1,0].scatter(data1[:, 0], data1[:, 1], c=color, s=50, cmap='viridis')
    axs[1,0].scatter(data1[:, 0], data1[:, 1],
                 facecolors='none', edgecolors=colorp, linestyle=style, s=130, linewidth=1.6)
    axs[1,0].set_xlabel('Peak [$\mu$V/mA]')
    axs[1,0].set_ylabel('PCI [#/mA]')
    axs[1,0].axis('square')
    log_reg = LogisticRegression(penalty = None, random_state = 1)
    cv_accuracy = np.mean(cross_val_score(log_reg, data1, y_roc, cv=LeaveOneOut()))
    log_reg.fit(data1, y_roc)
    w1, w2 = log_reg.coef_[0]
    b = log_reg.intercept_[0]
    print(w1,w2,b)
        
    x_min, x_max = data1[:, 0].min() - 1, data1[:, 0].max() + 1
    y_min, y_max = data1[:, 1].min() - 1, data1[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    axs[1,0].contourf(xx, yy, Z, cmap=cmap_light, alpha = 0.3)
    accuracy = accuracy_score(y_roc, log_reg.predict(data1[:,:2]))
    axs[1,0].set_title(f'NNS\nTrain: {100*accuracy:.2f}%\nLOSO validation: {100*cv_accuracy:.2f}%')
    
    # pdb.set_trace()

    logo = LeaveOneGroupOut()
    y_preds = np.empty_like(y_roc)
    subjects = np.random.randint(0, data1.shape[0], size=data1.shape[0])
    for train_index, test_index in logo.split(data1, y_roc, groups=subjects):
        X_train, X_test = data1[train_index], data1[test_index]
        y_train = y_roc[train_index]
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X_train, y_train)
        y_preds[test_index] = model.predict(X_test)
    cm = confusion_matrix(y_roc, y_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar = False, annot_kws = {'fontsize':15}, 
                ax=axs[2, 0])
    axs[2, 0].set_xlabel('Predicted', fontsize = 11)
    axs[2, 0].set_ylabel('True')
    axs[2, 0].set_aspect('equal')
    axs[2, 0].set_yticklabels(['cUWS','rUWS/MCS'], fontsize = 9)
    axs[2, 0].set_xticklabels(['cUWS','rUWS/MCS'], fontsize = 9)


    # Second clustering, SCS
    # XY plane
    axs[0,1].scatter(data2[:, 0], data2[:, 1], c=color, s=50, cmap='viridis')
    axs[0,1].scatter(data2[:, 0], data2[:, 1],
                 facecolors='none', edgecolors=colorp, linestyle=style, s=130, linewidth=1.6)
    axs[0,1].set_xlabel('Peak [$\mu$V/mA]')
    axs[0,1].set_ylabel('PCI [#/mA]')
    axs[0,1].set_title('SCS', fontsize = 12)
    axs[0,1].axis('square')

    # XZ plane
    axs[1,1].scatter(data2[:, 0], data2[:, 1], c=color, s=50, cmap='viridis')
    axs[1,1].scatter(data2[:, 0], data2[:, 1],
                 facecolors='none', edgecolors=colorp, linestyle=style, s=130, linewidth=1.6)
    axs[1,1].set_xlabel('Peak [$\mu$V/mA]')
    axs[1,1].set_ylabel('PCI [#/mA]')
    axs[1,1].set_title('SCS', fontsize = 12)
    axs[1,1].axis('square')
    log_reg = LogisticRegression(penalty = None,
                            random_state = 42)
    cv_accuracy = np.mean(cross_val_score(log_reg, data2[:,:2], y_roc, cv=LeaveOneOut()))
    log_reg.fit(data2[:,:2], y_roc)
    x_min, x_max = data2[:, 0].min() - 1, data2[:, 0].max() + 1
    y_min, y_max = data2[:, 1].min() - 1, data2[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    axs[1,1].contourf(xx, yy, Z, cmap=cmap_light, alpha = 0.3)
    accuracy = accuracy_score(y_roc, log_reg.predict(data2[:,:2]))
    axs[1,1].set_title(f'SCS\nTrain: {100*accuracy:.2f}%\nLOSO validation: {100*cv_accuracy:.2f}%')
    
    
    logo = LeaveOneGroupOut()
    y_preds = np.empty_like(y_roc)
    subjects = np.random.randint(0, data2.shape[0], size=data2.shape[0])
    for train_index, test_index in logo.split(data2, y_roc, groups=subjects):
        X_train, X_test = data2[train_index], data2[test_index]
        y_train = y_roc[train_index]
        model = LogisticRegression(solver='lbfgs', max_iter=1000)
        model.fit(X_train, y_train)
        y_preds[test_index] = model.predict(X_test)
    cm = confusion_matrix(y_roc, y_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar = False, annot_kws = {'fontsize':15}, 
                ax=axs[2, 1])
    axs[2, 1].set_xlabel('Predicted', fontsize = 11)
    axs[2, 1].set_ylabel('True')
    axs[2, 1].set_aspect('equal')
    axs[2, 1].set_yticklabels(['cUWS','rUWS/MCS'], fontsize = 9)
    axs[2, 1].set_xticklabels(['cUWS','rUWS/MCS'], fontsize = 9)
    

    fig.set_size_inches(18, 10, forward=True)
    fig.tight_layout()
    fig.savefig(os.path.join(opt_ex['fig_dir'], 'Figure_4_Bivariate.svg'), format='svg')
    return 'Success'

# %% Supplementary functions

def p_a_fig_s1(paired_features, opt_ex):
    
    amperes, amperes_sep = opt_ex['nns_thres'], opt_ex['scs_thres']   
    status, statusp, color, colorp, style = opt_ex['diagnosis_level'], opt_ex['prognosis_level'], opt_ex['color_diagnosis'], opt_ex['color_improvement'], opt_ex['style_improvement']
    improvement = [sp-s for sp, s in zip(statusp, status)]
    improvement = np.ravel([1 if imp > 0 else 0 for imp in improvement])


     
    df_plot_nns = {'PSD':[],'Status':[],'Band':[]}
    df_plot_scs = {'PSD':[],'Status':[],'Band':[]}

    for idx, (features_nns, features_scs) in enumerate(paired_features):
        for ixk, k in enumerate(opt_ex['eeg_bands'].keys()):
            if ixk >= 0:
                # pdb.set_trace()
                df_plot_nns['PSD'].append(100*features_nns['PSD '+k]/(features_nns['PSD delta']+features_nns['PSD theta']+features_nns['PSD alpha']+features_nns['PSD beta']))
                df_plot_nns['Status'].append(improvement[idx])
                df_plot_nns['Band'].append(ixk)
        
                df_plot_scs['PSD'].append(100*features_scs['PSD '+k]/(features_scs['PSD delta']+features_scs['PSD theta']+features_scs['PSD alpha']+features_scs['PSD beta']))
                df_plot_scs['Status'].append(improvement[idx])
                df_plot_scs['Band'].append(ixk)
    fig, axs = plt.subplots(2, 1, figsize=(7,5.5), sharey=True)
    
    sns.violinplot(data=df_plot_nns, x='Band', y='PSD', hue='Status', ax=axs[0], palette='twilight_shifted')
    # pdb.set_trace()
    # for i, band in enumerate([r'$\delta$', r'$\theta$', r'$\alpha$', r'$\beta$']):
    #     data_nns_band = [df_plot_nns['PSD'][j] for j in range(len(df_plot_nns['PSD'])) if df_plot_nns['Band'][j] == i]
    #     status_nns_band = [df_plot_nns['Status'][j] for j in range(len(df_plot_nns['Status'])) if df_plot_nns['Band'][j] == i]
    #     data_scs_band = [df_plot_scs['PSD'][j] for j in range(len(df_plot_scs['PSD'])) if df_plot_scs['Band'][j] == i]
    #     status_scs_band = [df_plot_scs['Status'][j] for j in range(len(df_plot_scs['Status'])) if df_plot_scs['Band'][j] == i]

    #     # Perform Mann-Whitney U test
    #     stat_nns, p_nns = mannwhitneyu([data_nns_band[j] for j in range(len(data_nns_band)) if status_nns_band[j] == 0],
    #                                    [data_nns_band[j] for j in range(len(data_nns_band)) if status_nns_band[j] == 1])
    #     stat_scs, p_scs = mannwhitneyu([data_scs_band[j] for j in range(len(data_scs_band)) if status_scs_band[j] == 0],
    #                                    [data_scs_band[j] for j in range(len(data_scs_band)) if status_scs_band[j] == 1])

        # Add annotations
        # axs[0].annotate(f'p={p_nns:.3f}', xy=(i, max(data_nns_band) + 5), ha='center', fontsize=10)
        # axs[1].annotate(f'p={p_scs:.3f}', xy=(i, max(data_scs_band) + 5), ha='center', fontsize=10)
    add_stat_annotation(axs[0], data=df_plot_nns, x='Band', y='PSD', hue='Status',
                        box_pairs=[((i, 0), (i, 1)) for i in range(4)],
                        test='Mann-Whitney', text_format='star', loc='outside', verbose=2, comparisons_correction = None)
    axs[0].set_title('NNS')
    axs[0].set_xlabel('Frequency band', fontsize = 12)
    axs[0].set_ylabel('Normalized Absolute\n Power [%]')
    axs[0].set_xticklabels([r'$\delta$',r'$\theta$',r'$\alpha$',r'$\beta$'])

    sns.violinplot(data=df_plot_scs, x='Band', y='PSD', hue='Status', ax=axs[1], palette='twilight_shifted')
    add_stat_annotation(axs[1], data=df_plot_scs, x='Band', y='PSD', hue='Status',
                        box_pairs=[((i, 0), (i, 1)) for i in range(4)],
                        test='Mann-Whitney', text_format='star', loc='outside', verbose=2, comparisons_correction = None)
    axs[1].set_title('SCS')
    axs[1].set_xlabel('Frequency band', fontsize = 12)
    axs[1].set_ylabel('Normalized Absolute\n Power [%]')
    axs[1].set_xticklabels([r'$\delta$',r'$\theta$',r'$\alpha$',r'$\beta$'])
    legend = plt.legend()
    legend.get_texts()[0].set_text("Not improved")  
    legend.get_texts()[1].set_text("Improved") 
    
    for ixax, ax in enumerate(axs):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
     
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(opt_ex['fig_dir'], "Figure_S1.svg"), format = 'svg')
    pd.DataFrame.from_dict(df_plot_nns).to_excel('psdnns.xlsx')
    pd.DataFrame.from_dict(df_plot_scs).to_excel('psdscs.xlsx')
    return 'Success'


def p_a_fig_s2(paired_features, opt_ex):
    amperes, amperes_sep = opt_ex['nns_thres'], opt_ex['scs_thres']   
    status, statusp, color, colorp, style = opt_ex['diagnosis_level'], opt_ex['prognosis_level'], opt_ex['color_diagnosis'], opt_ex['color_improvement'], opt_ex['style_improvement']
    improvement = [sp-s for sp, s in zip(statusp, status)]
    improvement = np.ravel([1 if imp > 0 else 0 for imp in improvement])
    
    
    times = np.linspace(0, 1, 512)
    rank_dict = {'NNS':[],'SCS':[],'Status':improvement}
    for idx, (features_nns, features_scs) in enumerate(paired_features):
        rank_dict['NNS'].append(features_nns['Rank'])
        rank_dict['SCS'].append(features_scs['Rank'])
    ranks_dict_paired = pd.DataFrame.from_dict({'Ranks': np.hstack([rank_dict['NNS'], rank_dict['SCS']]),
                                                'Subject': np.hstack([range(len(rank_dict['NNS'])), range(len(rank_dict['NNS']))]),
                                                'Within': np.hstack([[0]*len(rank_dict['NNS']), [1]*len(rank_dict['NNS'])])})
    rank_dict['NNS'] = np.array(rank_dict['NNS'])
    rank_dict['SCS'] = np.array(rank_dict['SCS'])
    rank_dict['Status'] = np.array(rank_dict['Status'])

    fig, axs = plt.subplots(1,3, figsize = (8,3), sharey = True)
    sns.violinplot(data = rank_dict, y = 'NNS', 
                x = 'Status', ax = axs[0],
                palette = 'twilight_shifted')       
     
    axs[0].set_xlabel('')
    axs[0].set_ylabel('EEG channels\nmean rank')
    axs[0].set_xticklabels(['Not improved', 'Improved'])
    axs[0].set_title('NNS')

    sns.violinplot(data = rank_dict, y = 'SCS', 
                x = 'Status', ax = axs[1],
                palette = 'twilight_shifted')
    axs[1].set_xlabel('')
    axs[1].set_ylabel('')
    axs[1].set_title('SCS')
    axs[1].set_xticklabels(['Not improved', 'Improved'])
    pg.plot_paired(data=ranks_dict_paired, dv='Ranks', within='Within', subject='Subject',
                   ax = axs[2])
    axs[2].set_xlabel('')
    axs[2].set_xticklabels(['NNS', 'SCS'])
    for ixax, ax in enumerate(axs):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if ixax > 0:
            # ax.set_yticks([])
            ax.spines['left'].set_visible(False)
            
    fig.tight_layout()
    fig.savefig(os.path.join(opt_ex['fig_dir'], "Figure_S2.svg"), format = 'svg')
    pdb.set_trace()
    from scipy.stats import wilcoxon
    print('NNS Rank')
    mannwhitneyu(rank_dict['NNS'][rank_dict['Status'] == 0], rank_dict['NNS'][rank_dict['Status'] == 1])
    print('SCS Rank')
    mannwhitneyu(rank_dict['SCS'][rank_dict['Status'] == 0], rank_dict['SCS'][rank_dict['Status'] == 1])
    stat, p_value = wilcoxon(rank_dict['NNS'], rank_dict['SCS'])
    print('Paired rank comparison')
    print(f'Wilcoxon statistic: {stat}')
    print(f'P-value: {p_value}')
    
    return 'Success'


#%% Execution
compute = 1
if compute:
    os.chdir(data_dir)
    with open("nns_l.pkl", 'rb') as file:
        data_nns = pickle.load(file)

    with open("scs_l.pkl", 'rb') as file:
        data_scs = pickle.load(file)
        
    process_EEG_tb(data_nns, opt_ex, 1)
    process_EEG_tb(data_scs, opt_ex, 0)
else:
    pass

paired_features = load_features(res_dir)
p_a_fig2(paired_features, opt_ex)
sd12 = p_a_fig3(paired_features, opt_ex)
opt_ex['sd1_over_sd2'] = sd12
p_a_fig4(paired_features, opt_ex)
p_a_fig_s1(paired_features, opt_ex)  
p_a_fig_s2(paired_features, opt_ex) 


