import h5py
import numpy as np
from functools import partial

def extract_dataset(file, dataset):
    if dataset.dtype == np.float64:
        #double
        #return extract_double_dataset(file, dataset)
        return dataset.value

    elif dataset.dtype == np.dtype:
        #cell
        return extract_cell_dataset(file, dataset)

    elif dataset.dtype == np.uint16:
        #string
        return extract_string(dataset)

def extract_cell_dataset(file, dataset):
    '''
    behold the elegance and simplicity of recursion
    '''
    return map_ndarray(
        partial(extract_dataset, file),
        np.array(map(
                    partial(map_ndarray, lambda y:file[y]),
                    dataset.value)))

def extract_string(dataset):
    #return map_ndarray(chr, dataset.value)
    #map_ndarray( reduce( lambda x,y:chr(x)+chr(y), dataset.value) )
    
    #return map_ndarray( reduce( lambda x,y:chr(x)+chr(y), np.array(dataset) ))

    string_k = partial(map_ndarrays, lambda z: ''.join(map(chr,z)))

#                    lambda z: reduce( lambda x,y:chr(x)+chr(y),
 #                                     map( int, z ) ))

    #string_k = partial(map_ndarrays, 
    #                partial(reduce, lambda x,y:chr(x)+chr(y)))

    #string_k = partial(map_ndarray, 
    #    lambda z:reduce( 
    #                lambda x,y:chr(x)+chr(y), 
    #                z))
    
    import pdb
    pdb.set_trace()
    return string_k(dataset.value)


def map_ndarray( callable, ndarray ):
    '''
    like map, but operates on every element of n-dimensional np.ndarray
    '''
    if ndarray.ndim > 1:
        #return np.array(map( partial(map_ndarray, callable), ndarray ))
        return map( partial(map_ndarray, callable), ndarray )
    else:
        #return np.array(map( callable, ndarray ))
        return map( callable, ndarray )

def map_ndarrays( callable, ndarray ):
    '''
    like map, but operates on every lowest-dim list of n-dimensional np.ndarray
    '''
    if ndarray.ndim > 1:
        #return np.array(map( partial(map_ndarray, callable), ndarray ))
        return map( partial(map_ndarrays, callable), ndarray )
    else:
        #return np.array(map( callable, ndarray ))
        return callable( ndarray )
