import numpy as np
import matplotlib.pyplot as plt


# get all ori base filters ------------------


def get_all_ori_base_filters(abs_boundaries, size, max_rank, sigma=1, n_ori=4):
    x = np.linspace(-abs_boundaries, abs_boundaries, num=size, dtype=np.int)
    X, Y = np.meshgrid(x, x)
    psi_list = list()
    m_list = list()
    for k in range(max_rank + 1):
        psi_, m_, _, _ = get_k_rank_filters(X, Y, k, sigma)
        psi_list = psi_list + psi_
        m_list = m_list + m_
    PSI = get_all_possible_rotations(psi_list, m_list, n_ori)
    return np.real(PSI)


# Useful functions -------------------
def fct(x):
    return np.math.factorial(x)


# Transformation between set of coordinates
def cartesian_to_polar(x, y):
    return np.sqrt(x ** 2 + y ** 2), np.arctan2(x, y)


def polar_to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


# 2D Hermite filters -----------------

# Functions to get single filters
def coeff_func(p, m, n):
    # compute coefficient useful for P function
    c = (-2) ** (n - p)
    num = fct(m + n) * fct(n)
    den = fct(m + p) * fct(p) * fct(n - p)
    coeff = c * num / den
    return coeff


def P(x, m, n):
    # Polynomial function useful for A_radial
    Pol = np.sum([coeff_func(p, m, n) * (x ** p) for p in np.arange(n + 1)], axis=0)
    return Pol


def A_radial(r, m=0, n=0, sigma=1):
    # compute radial dependence of filter
    return (
        ((r / sigma) ** m)
        * P((r / sigma) ** 2, m, n)
        * np.exp(-((r / (2 * sigma)) ** 2))
    )


def A1(r, theta, m=0, n=0, sigma=1, K=1):
    # compute filter associated to cosine
    ang_part = (K / sigma) * np.exp(1j * m * theta)
    rad_part = A_radial(r, m, n, sigma)
    return rad_part * ang_part


def A2(r, theta, m=0, n=0, sigma=1, K=1):
    # compute filter associated to sine
    ang_part = (K / sigma) * np.exp(1j * (m * theta - np.pi / 2))
    rad_part = A_radial(r, m, n, sigma)
    return rad_part * ang_part


#
def r2mn(k):
    # get m n index from rank k
    n_list = np.arange((k + 1) / 2)
    mn_list = [[k - 2 * n, n] for n in n_list]
    return mn_list


def get_k_rank_filters(X, Y, k, sigma=1):
    # return list of 2d Hermitian filters of rank k
    # defined on meshgrid X, Y, and list of corresponding m, n, sc values
    r, theta = cartesian_to_polar(X, Y)
    mn_list = r2mn(k)
    psi_list = []
    m_list = []
    n_list = []
    sc_list = []
    for mn in mn_list:
        m = mn[0]
        n = mn[1]
        psi_list.append(A1(r, theta, m, n, sigma))
        m_list.append(m)
        n_list.append(n)
        sc_list.append(0)
        if m != 0:
            psi_list.append(A2(r, theta, m, n, sigma))
            m_list.append(m)
            n_list.append(n)
            sc_list.append(1)
    return psi_list, m_list, n_list, sc_list


def max_r2Nf(max_r):
    Nf = 0
    for r in range(max_r + 1):
        Nf += r + 1
    return Nf


# rotate list of filters by selected angle. n.b. it requires the list of the m index corresponding to each filter!
def rotate_filters(psi_list, m_list, angle=np.pi / 4):
    psi_rotated_list = list()
    for psi, m in zip(psi_list, m_list):
        psi_rotated = np.exp(1j * m * angle) * psi
        psi_rotated_list.append(psi_rotated)
    return np.array(psi_rotated_list)


def get_all_possible_rotations(psi_list, m_list, n_ori):
    oris = 2 * np.arange(n_ori) * np.pi / n_ori
    PSI = list()
    for ori in oris:
        PSI.append(np.array(rotate_filters(psi_list, m_list, ori)))
    return np.array(PSI)


# plotting functions
def plot_k_rank_filters(X, Y, k, sigma=1):
    print("rank", k, "2dHermite filters")
    psi_list, _, _, _ = get_k_rank_filters(X, Y, k, sigma)
    l = len(psi_list)
    for i in range(l):
        plt.colorbar()
        plt.axis("square")
        plt.show()
    return


def plot_up2k_rank_filters(X, Y, k, sigma=1):
    for i in range(k + 1):
        plot_k_rank_filters(X, Y, i, sigma)
    return


# abs_boundaries = 9
# size = 20
# max_rank = 4
# f = get_all_ori_base_filters(abs_boundaries, size, max_rank)
# _, ax= plt.subplots(4, len(f[0]))

# for ori in range(4):
#     for i in range(len(f[0])):
#         ax[ori, i].imshow(f[ori ,i], cmap='Greys_r')
# plt.show()