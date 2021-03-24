import numpy as np

def get_all_ori_base_filters(size):
    filters = get_list_base_filters(size)
    all_ori_base_filters = np.stack(
        [rotate_base_filter(filters, times) for times in range(4)]
    )
    return all_ori_base_filters


def rotate_base_filter(filters, num_times=1):
    for n in range(num_times):
        filters = np.array([rotate(f) for f in filters])
    return filters


def get_list_base_filters(size):
    one_then_zeros = np.zeros(size * size)
    one_then_zeros[0] = 1
    filters = [
        np.roll(one_then_zeros, i).reshape(size, size) for i in range(size * size)
    ]
    return np.array(filters)


def rotate(A):
    B = np.flip(A.transpose(1, 0), 1)
    return B