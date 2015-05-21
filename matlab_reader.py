import h5py
import numpy as np
from functools import partial

def extract_dataset(file, dataset):

    print dataset

    if dataset.dtype == np.float64:

        return extract_double_dataset(file, dataset)

    elif dataset.dtype == np.dtype:

        #print 'wiscasset'
        #print dataset
        #import pdb
        #pdb.set_trace()

        return extract_cell_dataset(file, dataset)

    elif dataset.dtype == np.uint16:
        return extract_string(dataset)

def extract_double_dataset(file, dataset):
    return dataset.value 

def extract_cell_dataset(file, dataset):
    ndarray = dataset.value

    return map_ndarray(
        partial(extract_dataset, file),
        np.array(map(
                    partial(map_ndarray, lambda y:file[y]),
                    dataset.value)))

#    return map_ndarray(
#        lambda z:extract_dataset(file, z),
#            np.array(
#                map(
#                    lambda x:map_ndarray(
#                                          lambda y:file[y],
#                                          x 
#                                        ),
#                    dataset.value)
#            )
#    )

    dataset_array = np.array( 

        map( partial(map_ndarray, lambda y:file[y]), ndarray) )

        #map(lambda x:map_cell_recursive(file, x), ndarray) )

#    #return map(lambda x:extract_dataset(file, x) )
    print dataset_array

    return map_ndarray( 
        lambda z:extract_dataset(file, z), 
        dataset_array
    )

#def map_cell_recursive(file, ndarray):
#    if ndarray.ndim > 1:
#        return map(lambda x:map_cell_recursive(file, x), ndarray)
#    else:
#        #extract item from cell
#        return map( lambda x:file[x], ndarray)

def extract_string(dataset):
    ndarray = dataset.value
    return map_ndarray(chr, dataset)
    return map(map_char_recursive, dataset)

#def map_char_recursive(ndarray):
#    if ndarray.ndim > 1:
#        return map(map_char_recursive, ndarray)
#    else:
#        map(chr, ndarray)

def map_ndarray( callable, ndarray ):
    if ndarray.ndim > 1:
        return map( partial(map_ndarray, callable), ndarray )
    else:
        return map( callable, ndarray )
