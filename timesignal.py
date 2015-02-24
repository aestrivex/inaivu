from traits.api import HasTraits, Float, Dict, Int
import numpy as np

class InvasiveSignal(HasTraits):

    sampling_rate = Float
    signals_dict = Dict # Str -> np.1darray
    length = Int

def gen_random_invasive_signal(ch_names, length):
    
    ivs = InvasiveSignal()
    ivs.sampling_rate = 1
    ivs.length = length

    tsignal = {}
    for ch_name in ch_names:
        tsignal[ch_name] = np.random.random(length)
    return tsignal

def gen_stupid_gamma_signal(ch_names):

    ivs = InvasiveSignal()
    ivs.sampling_rate = 1
    ivs.length = 100

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
