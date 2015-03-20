from __future__ import division

from traits.api import HasTraits, Float, Dict, Int, Any, List
import numpy as np
from scipy import io
import mne

class SourceSignal(HasTraits):
    mne_source_estimate = Any #Instance(mne._BaseSourceEstimate)

class NoninvasiveSignal(SourceSignal):
    pass

class InvasiveSignal(SourceSignal):
    mne_source_estimate = Any #Instance(mne._BaseSourceEstimate)
    ix_pos_map = Any #Instance(np.ndarray)
    ch_names = List

def load_stc(stcf):
    return mne.read_source_estimate(stcf)

def signal_from_stc(stc, ordering=None, invasive=False):
    ivs = InvasiveSignal()
    ivs.mne_source_estimate = stc

    if ordering is not None:
        ivs.ix_pos_map, ivs.ch_names = load_ordering_file(ordering)
        
    return ivs

def load_montage(montage_file):
    return mne.channels.read_montage(os.path.realpath(montage_file))

def load_ordering_file(ordering_file):
    ixes = []
    names = []

    with open(ordering_file) as fd:
        for i,ln in enumerate(fd):
            if ln=='delete':
                continue
            ixes.append(i)
            names.append(ln)

    return ixes, names
    
def save_ordering_file(fname, ordering):
    with open(ordering, 'w') as fd:
        for name in ordering:
            fd.write('%s\n'%name)

def gen_stupid_gamma_signal(ch_names, hemi='rh'):

    ivs = InvasiveSignal()
    ivs.ch_names = ch_names

    if hemi=='rh':
        vertnos = [np.array(()), np.arange(len(ch_names))]
    else:
        vertnos = [np.arange(len(ch_names)), np.array()]

    from scipy.stats import gamma

    tsignal = np.zeros((len(ch_names), 100))
    for i,ch_name in enumerate(ch_names):
        scale_param = np.random.randint(5,15)
        multi_param = np.random.randint(1,4)
        min_param = np.random.randint(10)
        max_param = np.random.randint(10,15)

        #import pdb
        #pdb.set_trace()

        tsignal[i,:] = np.array([gamma.pdf(j,scale_param)*multi_param for
            j in np.linspace(min_param,max_param,100)])

    stc = mne.SourceEstimate(tsignal, tmin=1, tstep=1, vertices=vertnos,
        subject='woethiezluok')

    ivs.mne_source_estimate = stc

    return ivs

def stc_from_fiff(fiff_file, names=None):
    '''
    Creates a source estimate from the sensor space channels in a fiff file
    '''
    ra = mne.io.Raw(fiff_file)

    tmin = ra.index_as_time(0)
    tstep = 1/ra.info['sfreq']

    if names is not None:
        vertnos = np.squeeze([np.where(np.array(ra.ch_names)==name) for name in
            np.intersect1d(ra.ch_names, names)])

    else:
        #use all names
        vertnos = np.arange(len(ra.ch_names))

    ordering_names = np.array(ra.ch_names)
    ordering_names[ np.setdiff1d( np.arange(len(ra.ch_names)), vertnos)] = 'del'

    #convert to 'delete' without messing with datatypes
    ordering_names = np.array(map(lambda x:'delete' if x=='del' else x,
        ordering_names.tolist()))

    #by arbitrary choice we use the RH no matter where are the electrodes
    stc = mne.SourceEstimate(ra[:][0][np.sort(vertnos)],
        vertices = [np.array(()), np.sort(vertnos)], tmin=tmin, tstep=tstep)

    return stc

# this function does not operate on well formulated fieldtrip files
# consequently it does no processing that we should trust

def create_signal_from_fieldtrip_timefqchan(ft_file):
    ftd = io.loadmat(ft_file)

    nr_frequencies = ftd['datad']['trial'][0,0].size

    channels = map(str, map(lambda x:x[0][0], ftd['ElectrodeLabels']))
    nr_channels = ftd['datad']['trial'][0,0][0,0].size

    for i in xrange(nr_frequencies):

        ivs = InvasiveSignal()

        ivs.sampling_rate = float(ftd['datad']['fsample']) 

        ivs.time_start = float(ftd['datad'][0,0][0][0][0][0])
        ivs.time_end = float(ftd['datad'][0,0][0][0][-1][0])

        tsignal = {}
        
        for j, ch in enumerate(channels):
            tsignal[ch] = ftd['ChanFreqTimeMatrix'][j,i,:]
        
        ivs.signals_dict = tsignal

        yield ivs
