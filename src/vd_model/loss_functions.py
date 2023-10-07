import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial import distance
from scipy.stats import ks_2samp
from scipy.special import kl_div
import torch.nn.functional as F
from torch import nn
import torch


    
def kolomogorow_smirnow(train_dist, test_dist):
    ks_statistic, p_value = ks_2samp(train_dist, test_dist)
    return ks_statistic


def kl_divergence(p, q):
    """
    Compute KL divergence between two arrays representing samples from arbitrary distributions.
    p, q: Arrays representing the samples from the distributions.
    """
    # Compute histogram-based PDFs
    num_bins = min(int(np.sqrt(len(p))), 100)  # Choose an appropriate number of bins
    p_hist, p_bins = np.histogram(p, bins=num_bins, density=True)
    q_hist, q_bins = np.histogram(q, bins=num_bins, density=True)
    
    # Compute bin centers
    p_centers = 0.5 * (p_bins[1:] + p_bins[:-1])
    q_centers = 0.5 * (q_bins[1:] + q_bins[:-1])
    
    # Compute PDF values at bin centers
    p_pdf = np.interp(p_centers, p_bins[:-1], p_hist)
    q_pdf = np.interp(q_centers, q_bins[:-1], q_hist)
    
    # Clip values to avoid division by zero
    p_pdf = np.clip(p_pdf, 1e-10, None)
    q_pdf = np.clip(q_pdf, 1e-10, None)
    
    # Compute KL Divergence
    kl_div = np.sum(p_pdf * np.log(p_pdf / q_pdf))
    return kl_div * (np.mean(p) - np.mean(q)) ** 2

    
