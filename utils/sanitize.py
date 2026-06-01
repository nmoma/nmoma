import numpy as np
from utils.utils import *
import h5py


# max_joint = max_joint.numpy()
# min_joint = min_joint.numpy()

print(max_joint)
print(min_joint)
chunk_size = 62500

def main():
    f = h5py.File("train_nonfix.h5", 'r')

    wps = f['wps']
    print("wps shape: ", wps.shape)

    for i, lo in enumerate(range(0, wps.shape[0], chunk_size)):
        hi = min(lo+chunk_size, wps.shape[0])
        wps_chunk = wps[lo:hi]

        norm_wps = wps_chunk[..., -7:] / max_joint

        print(norm_wps.max())
        print(norm_wps.min())

        e = np.any(np.logical_or(norm_wps > 1.0, 
                                 norm_wps <-1.0))
        if e:
            print("Error in chunk ", i, ":", lo, ":", hi)
    f.close()


if __name__ == '__main__':
    main()