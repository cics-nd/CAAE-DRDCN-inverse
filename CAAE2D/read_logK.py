import numpy as np
import h5py
import scipy.io

def read_input(ndata, dx, ngx, ngy):
    x = np.full( (ndata,dx,ngx,ngy), 0.0)
    for i in range(1, ndata + 1):
        K = scipy.io.loadmat('input/cond{}.mat'.format(i))
        K = K['cond']
        x[i-1,0, :, :] = np.log(K)  # K is the first input channel, log transformed

    print("X: {}".format(x[0,]))
    hf = h5py.File('input_lhs{}.hdf5'.format(ndata), 'w')
    hf.create_dataset('dataset', data = x, dtype ='f', compression = 'gzip')
    hf.close()

ndata = 40000
dx = 1
ngx = 40
ngy = 80
read_input(ndata,dx,ngx,ngy)
