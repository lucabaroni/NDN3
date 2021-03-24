import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


def get_all_ori_base_filters(size=9, abs_boundaries=None, max_mu=None, max_k=None, sigma=0.75, n_ori=4):
    if abs_boundaries == None:
        abs_boundaries = int(size/2)
    if max_mu == None:
        max_mu = int(abs_boundaries-1)
    if max_k == None:
        max_k = int(size/2)
    x = np.linspace(-abs_boundaries, abs_boundaries, num=size, dtype=np.int)
    X, Y = np.meshgrid(x, x)
    r, theta = cartesian_to_polar(X, Y)
    filters, k_of_each_filter = get_all_filters(r, theta, max_mu, max_k, sigma)
    filters_all_ori = get_all_possible_rotations(filters, k_of_each_filter, n_ori)
    return filters_all_ori
    
def get_all_possible_rotations(filters, ks, n_ori):
    angles = np.linspace(0, 2*np.pi, n_ori+1)[1 :-1]
    n_filters = len(filters) 
    all_ori_filters = [deepcopy(filters)]
    for angle in angles:
        filters_r = [rotate_filter(filters[i], ks[i], angle) for i in range(n_filters)] 
        all_ori_filters = np.vstack((all_ori_filters, [filters_r]))
    return np.real(all_ori_filters)

def get_all_filters(r, theta, max_mu, max_k, sigma=1):
    k_list = np.arange(0, max_k+1).tolist()
    mu_list = np.arange(1, max_mu+1).tolist()
    filters = [get_filter(r, theta, mu, k, sigma) for mu in mu_list for k in k_list]
    
    filter_0 = [get_filter(r, theta, 0, 0, sigma)]
    filters = filter_0 + filters
    k_of_each_filter = [0] + k_list*max_mu
    return filters, k_of_each_filter    

def get_filter(r, theta, mu, k, sigma=1):
    radial_term = np.exp(-(r-mu)**2/(2*sigma**2))
    angular_term = np.exp(1j*k*(theta - np.pi/2))
    return radial_term*angular_term

def rotate_filter(filter, k, angle):
    return np.array(filter)*np.exp(1j*k*angle)

def cartesian_to_polar(x, y):
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(x, y)

sigmas = [0.6, 0.8]
size = 9
n_ori = 4
f = get_all_ori_base_filters(size=size)
_, ax= plt.subplots(len(sigmas), len(f[0]))

for j in range(len(sigmas)):
    for i in range(len(f[0])):
        f = get_all_ori_base_filters(size = size, sigma=sigmas[j])
        ax[j, i].imshow(f[0 ,i], cmap='Greys_r', vmin=-1, vmax=1)
plt.show()

#TODO check that global rotation is consistent with rotation of filters
