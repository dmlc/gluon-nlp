import pickle
import mmap

if pickle.HIGHEST_PROTOCOL < 5:
    del pickle
    import pickle5 as pickle


def serialize(path, tbl):
    """Serialize tbl with out-of-band data to path for zero-copy shared memory usage.

    If the object to be serialized itself, or the objects it uses for data
    storage (such as numpy arrays) implement the the pickle protocol version 5
    pickle.PickleBuffer type in __reduce_ex__, then this function can store
    these buffers out-of-band as files in `path` so that they subsequently be
    re-used for zero-copy sharing accross processes.

    Parameters
    ----------
    path : pathlib.Path
        Empty folder used to save serialized data. Usually a folder /dev/shm
    tbl : object
        Object to serialize. For example a PyArrow Table, a Pandas Dataframe or
        any type that relies on NumPy to store the binary data.

    """
    idx = 0

    def buffer_callback(buf):
        nonlocal idx
        with open(path / f'{idx}.bin', 'wb') as f:
            f.write(buf)
        idx += 1

    with open(path / 'meta.pkl', 'wb') as f:
        pickle.dump(tbl, f, protocol=5, buffer_callback=buffer_callback)


def load(path):
    """Load serialized object with out-of-band data from path based on zero-copy shared memory.

    Parameters
    ----------
    path : pathlib.Path
        Folder used to save serialized data with serialize(). Usually a folder /dev/shm

    """
    num_buffers = len(list(path.iterdir())) - 1  # exclude meta.idx
    buffers = []
    for idx in range(num_buffers):
        f = open(path / f'{idx}.bin', 'rb')
        buffers.append(mmap.mmap(f.fileno(), 0, prot=mmap.PROT_READ))
    with open(path / 'meta.pkl', 'rb') as f:
        return pickle.load(f, buffers=buffers)
