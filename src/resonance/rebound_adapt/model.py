from ..neuralnodes import utils, NL, func
import numpy as np
from .params import Params



def run_model(stim: np.ndarray, p: Params, fs: float = 1_000, verbose_output: bool = False, return_dict: bool = False):
    # --- STIMULUS -----------------------------------------------------------
    try:
        p = Params.from_list(p)
    except:
        pass

    p.filter_delay /= (1000 / fs)
    p.filter_inh_dur /= (1000 / fs)
    p.filter_exc_dur /= (1000 / fs)
    # rebound neuron
    filt_pos = p.filter_exc_gain * np.ones((int(p.filter_exc_dur),))
    filt_neg = -p.filter_inh_gain * np.ones((int(p.filter_inh_dur),))
    filter = np.concatenate((filt_pos, filt_neg), axis=0)
    sub_output_linear = func.ln(-stim, filter)
    #sub_output = NL.relu(sub_output_linear)
    sub_output = np.clip(sub_output_linear,0,1)

    # delayed relay neuron
    stim_delay = func.synapse(stim, p.filter_delay)

    # coincidence detector
    output = np.multiply(stim_delay, sub_output)

    if return_dict:
        out = {'stim': stim, 'an1': output, 'p': p}
    else:
        if verbose_output:
            out = (stim, sub_output_linear, sub_output, stim_delay, output, p)
        else:
            out = (output,)
    return out


if __name__ == "__main__":
    # from rebound_params import Params

    input_stim, s, ppau, pdur = utils.makePPFstim()

    out = run_model(input_stim, p)
