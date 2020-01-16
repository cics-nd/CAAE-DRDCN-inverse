import numpy as np
import h5py
import sys
import scipy.io
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

plt.switch_backend('agg')


def plot(idx):
    x = scipy.io.loadmat('input/{}.mat'.format(idx))
    x = x['cond']
    nrow = x.shape[0]//2
    fig, axes = plt.subplots(nrow,2)
    for j, ax in enumerate(fig.axes):
        if j < x.shape[0]:
            ax.set_axis_off()
            ax.set_aspect('equal')

            cax = ax.imshow(x[j], cmap='jet', origin='lower')

            cbar = plt.colorbar(cax, ax=ax, fraction=0.040, pad=0.04,
                                            format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((-2, 2))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()

    plt.savefig("{}.png".format(idx),bbox_inches='tight',dpi=300)

def crop(input, depth, height, width, stride_d, stride_h, stride_w,n):
    D, H, W = input.shape # height and width of the original image
    print(input.shape)
    i_z   = (D-depth)//stride_d  # maximum z index
    i_row = (H-height)//stride_h # maximum row index
    i_col = (W-width)//stride_w  # maximum column index

    print(i_z,i_row,i_col)
    nn = 1
    for i in range(i_z): # along z-axis first
        for j in range(i_row): # horizontally second
            for k in range(i_col): # then vertically
                # print(i,j)
                cond = input[i*stride_d : i*stride_d+depth, j*stride_h:j*stride_h+height, k*stride_w : k*stride_w+width]
                n += 1
                scipy.io.savemat('input/{}.mat'.format(n), dict(cond=cond))  # save results
    print(n)
    return n


def read_input(ndata, dx, depth, height, width):
    x = np.full( (ndata,dx,depth, height, width), 0.0)
    for i in range(1, ndata + 1):
        K = scipy.io.loadmat('input/{}.mat'.format(i))
        x[i-1,0, :, :, :] = K['cond']

    print("X: {}".format(x[0,]))
    hf = h5py.File('input_lhs{}.hdf5'.format(ndata), 'w')
    hf.create_dataset('dataset', data = x, dtype ='f', compression = 'gzip')
    hf.close()

depth  = 6
height = 32
width  = 64
stride_d, stride_h, stride_w = 2, 6, 10
n = 0
for i in range(1,5):
    im = scipy.io.loadmat('K{}.mat'.format(i))
    im = im['K'] # image size N x D x H x W
    n = crop(im, depth, height, width, stride_d, stride_h, stride_w,n)

# save the data
ndata = 3000
dx = 1
read_input(ndata,dx,ngz,ngx,ngy)
