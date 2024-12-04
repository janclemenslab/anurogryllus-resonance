from ..neuralnodes import utils, func
import numpy as np
from .params import Params


def run_model(stim: np.ndarray, p: Params, fs: float = 1_000, verbose_output: bool = False, return_dict: bool = False):
    # --- STIMULUS -----------------------------------------------------------
    try:
        p = Params.from_list(p)
    except:
        pass
    p.filter_delay /= 1000 / fs
    p.filter_inh_dur /= 1000 / fs
    p.filter_exc_dur /= 1000 / fs
    p.ln2_filter_exc_dur /= 1000 / fs
    p.ln2_filter_inh_dur /= 1000 / fs
    p.ln2_ln4_delay /= 1000 / fs
    # rebound neuron

    filt_pos = p.filter_exc_gain * np.ones((int(p.filter_exc_dur),))
    filt_neg = -p.filter_inh_gain * np.ones((int(p.filter_inh_dur),))
    filter = np.concatenate((filt_pos, filt_neg), axis=0)
    sub_output_linear = func.ln(-stim, filter)
    # sub_output = np.clip(sub_output_linear, 0, 1)
    sub_output = np.clip(sub_output_linear, 0, np.inf)  # JAN changed upper limit from 1 to inf to ensure this acts as a relu

    # delayed relay neuron
    stim_delay = func.synapse(stim, p.filter_delay)

    # coincidence detector
    ln3_output = np.multiply(stim_delay, sub_output)

    # suppression
    filt_pos = p.filter_exc_gain * np.ones((int(p.filter_exc_dur),))
    filt_neg = -p.filter_inh_gain * np.ones((int(p.filter_inh_dur),))
    p.ln2_filter = np.concatenate((filt_neg, filt_pos), axis=0)
    # p.ln2_nonlinearity = lambda x=None: np.clip(x, -1, 0)
    p.ln2_nonlinearity = lambda x=None: np.clip(x, -np.inf, 0)    # JAN changed lower limit from -1 to -inf to ensure this acts as a relu

    ln2_output = p.ln2_nonlinearity(func.ln(stim, p.ln2_filter))

    ln2_ln4_output = func.synapse(ln2_output, p.ln2_ln4_delay, p.ln2_ln4_gain)
    p.output_nonlinearity = lambda x=None: np.clip(x, 0, np.inf)  # JAN changed upper limit from 1 to inf to ensure this acts as a relu

    output = p.output_nonlinearity(func.summate([ln2_ln4_output, ln3_output]))

    if return_dict:
        out = {"stim": stim, "an1": output, "p": p}
    else:
        if verbose_output:
            out = (stim, sub_output_linear, stim_delay, ln3_output, ln2_output, ln2_ln4_output, output)
        else:
            out = (output,)
    return out


if __name__ == "__main__":
    # from rebound_params import Params
    p = Params()
    input_stim, s, ppau, pdur = utils.makePPFstim()

    out = run_model(input_stim, p)
    print(out)
