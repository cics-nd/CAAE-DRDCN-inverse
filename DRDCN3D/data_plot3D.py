import h5py
import torch as th
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
plt.switch_backend('agg')


def load_data(hdf5_dir, args, kwargs, flag):

    if flag == 'train':
        batch_size = args.batch_size
        with h5py.File(hdf5_dir + "/input_lhs{}.hdf5".format(4000), 'r') as f:
            x = f['dataset'][()]
        print("x: {}".format(x[0]))
        with h5py.File(hdf5_dir + "/output_lhs{}.hdf5".format(4000), 'r') as f:
            y = f['dataset'][()]
        x = x[:args.n_train]
        y = y[:args.n_train]

    elif flag == 'test':
        batch_size = args.test_batch_size
        with h5py.File(hdf5_dir + "/input_lhs{}.hdf5".format(1000), 'r') as f:
            x = f['dataset'][()]
        print("x: {}".format(x[0]))
        with h5py.File(hdf5_dir + "/output_lhs{}.hdf5".format(1000), 'r') as f:
            y = f['dataset'][()]

    y_var = np.sum((y - np.mean(y, 0)) ** 2)

    data = th.utils.data.TensorDataset(th.FloatTensor(x), th.FloatTensor(y))
    data_loader = th.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, **kwargs)
    n_out_pixels = len(data_loader.dataset) * data_loader.dataset[0][1].numel()

    print("total input data shape: {}".format(x.shape))
    print("total output data shape: {}".format(y.shape))

    return x, y, n_out_pixels, y_var, data_loader


def plot_pred(samples_target, samples_output, epoch, idx, t, output_dir):

    samples_err = samples_target - samples_output
    samples = np.vstack((samples_target, samples_output, samples_err))

    Nout = samples_target.shape[0]
    c_max = np.full( (Nout*3), 0.0)
    c_min = np.full( (Nout*3), 0.0)
    for l in range(Nout*3):
        if l < Nout:
            c_max[l] = np.max(samples[l])
        elif Nout <= l < 2*Nout:
            c_max[l] = np.max(samples[l])
            if c_max[l] > c_max[l-Nout]:
                c_max[l-Nout] = c_max[l]
            else:
                c_max[l] = c_max[l-Nout]
        else:
            c_max[l] = np.max( np.abs(samples[l]) )
            c_min[l] = 0. - np.max( np.abs(samples[l]) )

    LetterId = (['a','b','c','d', 'e','f','g','h', 'i','j','k','m'])
    ylabel = (['$\mathbf{y}$', '$\hat{\mathbf{y}}$', '$\mathbf{y}-\hat{\mathbf{y}}$'])

    Ncol = Nout//2 # number of column in the subplot
    fig = plt.figure(figsize=(Ncol*4+0.5, 12))
    outer = gridspec.GridSpec(2, 1, wspace=0.01, hspace=0.08)
    m = 0
    samp_id = [ [0,1,2, 6,7,8, 12,13,14], [3,4,5, 9,10,11, 15,16,17] ]
    for j in range(2):
        inner = gridspec.GridSpecFromSubplotSpec(3, Ncol, subplot_spec = outer[j], wspace=0.1, hspace=0.16)
        l = 0
        for k in range(3*Ncol):
            ax = plt.Subplot(fig, inner[k])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            s_id = samp_id[j][k]
            cax = ax.imshow(samples[s_id], cmap='jet', origin='lower',vmin=c_min[s_id], vmax=c_max[s_id])
            fig.add_subplot(ax)
            ax.spines['left'].set_color('white')
            ax.spines['right'].set_color('white')
            ax.spines['bottom'].set_color('white')
            ax.spines['top'].set_color('white')

            cbar = plt.colorbar(cax, ax=ax, fraction=0.02, pad=0.015,
                        format=ticker.ScalarFormatter(useMathText=True))
            cbar.formatter.set_powerlimits((0, 0))
            cbar.ax.yaxis.set_offset_position('left')
            cbar.update_ticks()
            cbar.ax.tick_params(axis='both', which='both', length=0)
            cbar.ax.yaxis.get_offset_text().set_fontsize(12)
            cbar.ax.tick_params(labelsize=12)


            if k < Ncol:
                ax.text(22, 34, 'Layer {}'.format(m+1), fontsize=14,color='black')
                m = m + 1
            if np.mod(k,Ncol) == 0:
                if j == 0:
                    ax.set_ylabel(ylabel[l], fontsize=14)
                    l = 1 + l
                else:
                    ax.set_ylabel(ylabel[l], fontsize=14)
                    l = 1 + l

    plt.savefig(output_dir + '/epoch_{}_output_{}_oc{}.png'.format(epoch, idx, t),
                bbox_inches='tight',dpi=500) #,pad_inches = 0)
    plt.close(fig)
    print("epoch {}, done with printing sample output {}".format(epoch, idx))


def plot_r2_rmse(r2_train, r2_test, rmse_train, rmse_test, exp_dir, args):
    x = np.arange(args.log_interval, args.n_epochs + args.log_interval,
                args.log_interval)
    plt.figure()
    plt.plot(x, r2_train, 'k', label="train: {:.3f}".format(np.mean(r2_train[-5: -1])))
    plt.plot(x, r2_test, 'r', linestyle = '--', label="test: {:.3f}".format(np.mean(r2_test[-5: -1])))
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('$R^2$', fontsize=14)
    plt.legend(loc='lower right')
    plt.savefig(exp_dir + "/r2.png", dpi=400)
    plt.close()
    np.savetxt(exp_dir + "/r2_train.txt", r2_train)
    np.savetxt(exp_dir + "/r2_test.txt", r2_test)

    plt.figure()
    plt.plot(x, rmse_train, 'k', label="train: {:.3f}".format(np.mean(rmse_train[-5: -1])))
    plt.plot(x, rmse_test, 'r', linestyle = '--', label="test: {:.3f}".format(np.mean(rmse_test[-5: -1])))
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    plt.legend(loc='upper right')
    plt.savefig(exp_dir + "/rmse.png", dpi=400)
    plt.close()
    np.savetxt(exp_dir + "/rmse_train.txt", rmse_train)
    np.savetxt(exp_dir + "/rmse_test.txt", rmse_test)
