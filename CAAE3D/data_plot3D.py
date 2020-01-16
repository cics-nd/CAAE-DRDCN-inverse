import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torchvision.datasets as datasets
import torch as th
plt.switch_backend('agg')


def load_data(hdf5_dir, opt):

    with h5py.File(hdf5_dir + "train/input_lhs{}.hdf5".format(opt.n_train), 'r') as f:
        x_train = f['dataset'][()]

    with h5py.File(hdf5_dir + "test/input_lhs{}.hdf5".format(opt.n_test), 'r') as f:
        x_test = f['dataset'][()]

    print("total training data shape: {}".format(x_train.shape))

    data = th.utils.data.TensorDataset(th.FloatTensor(x_train))
    data_loader = th.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=True, num_workers=int(2))

    return data_loader, x_test


def plot_pred(samples, epoch, idx, output_dir):
    Ncol = 3
    Nrow = samples.shape[0] // Ncol

    fig, axes = plt.subplots(Nrow, Ncol, figsize=(Ncol*4, Nrow*2.1))
    fs = 16 # font size
    for j, ax in enumerate(fig.axes):
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        if j < samples.shape[0]:

            cax = ax.imshow(samples[j], cmap='jet', origin='lower', vmin=0, vmax=5)
            cbar = plt.colorbar(cax, ax=ax, fraction=0.025, pad=0.04,
                            format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()
            cbar.ax.tick_params(axis='both', which='both', length=0)
            cbar.ax.yaxis.get_offset_text().set_fontsize(fs-3)
            cbar.ax.tick_params(labelsize=fs-2)

    plt.savefig(output_dir+'/epoch_{}_{}.png'.format(epoch,idx), bbox_inches='tight',dpi=600)
    plt.close(fig)

    print("epoch {}, done printing".format(epoch))

