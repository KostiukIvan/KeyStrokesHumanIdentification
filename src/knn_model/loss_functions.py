from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.spatial import distance
from scipy.special import kl_div
from scipy.interpolate import interp1d
import numpy as np


def kolomogorow_smirnow(train_dist, test_dist):
    # Assuming data1 and data2 are your two distributions
    ks_statistic, p_value = ks_2samp(train_dist, test_dist)
    return ks_statistic, p_value

def t_test(train_dist, test_dist):
    # Assuming data1 and data2 are your two distributions
    t_statistic, p_value = ttest_ind(train_dist, test_dist)
    return t_statistic, p_value


def mann_whitney_u_test(train_dist, test_dist):
    # Assuming data1 and data2 are your two distributions
    u_statistic, p_value = mannwhitneyu(train_dist, test_dist)
    return u_statistic, p_value



def kl_divergense_non_symetric(train_dist, test_dist):
    inter_p, inter_q = _interpolation(train_dist, test_dist)
    # Compute KL Divergence from q to p
    kl_divergence = np.sum(np.where(inter_p != 0, inter_p * np.log(inter_p / inter_q), 0))
    return kl_divergence, None

def jennsen_shanon_symetric(train_dist, test_dist):
    inter_p, inter_q = _interpolation(train_dist, test_dist)
    # Compute JS Divergence
    js_divergence = distance.jensenshannon(inter_p, inter_q, 2.0)  # 2.0 represents the base of the logarithm

    return js_divergence, None


def _interpolation(train_dist, test_dist):
    p = np.array(train_dist)
    q = np.array(test_dist)

    # Assuming data1 and data2 are your distributions represented as arrays
    # Interpolate the data to create distributions with the same number of points
    interp_func1 = interp1d(range(len(p)), p, kind='linear')
    interp_func2 = interp1d(range(len(q)), q, kind='linear')

    # Generate new interpolated data with the same number of points
    if len(p) > len(q):
        inter_q = interp_func2(np.linspace(0, len(q) - 1, len(p)))
        inter_p = p
    else:
        inter_q = q
        inter_p = interp_func1(np.linspace(0, len(p) - 1, len(q)))

    return inter_p, inter_q

