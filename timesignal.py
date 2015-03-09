from traits.api import HasTraits, Float, Dict, Int
import numpy as np
from scipy import io

class InvasiveSignal(HasTraits):

    sampling_rate = Float
    signals_dict = Dict # Str -> np.1darray
    time_start = Float
    time_end = Float

def gen_stupid_gamma_signal(ch_names):

    ivs = InvasiveSignal()
    ivs.sampling_rate = 1
    ivs.time_start = 1
    ivs.time_end = 1

    from scipy.stats import gamma

    tsignal = {}
    for ch_name in ch_names:
        scale_param = np.random.randint(5,15)
        multi_param = np.random.randint(1,4)
        min_param = np.random.randint(10)
        max_param = np.random.randint(10,15)

        tsignal[ch_name] = [gamma.pdf(i,scale_param)*multi_param for
            i in np.linspace(min_param, max_param,100)]

    return tsignal

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
