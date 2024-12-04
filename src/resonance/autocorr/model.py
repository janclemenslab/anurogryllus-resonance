from ..neuralnodes import utils, NL, func
import numpy as np
from .params import Params
from typing import List, NamedTuple, Union


def run_model(stim: np.ndarray,
              p: Union[List, NamedTuple],
              fs: float = 1_000.0,
              verbose_output: bool = False,
              return_dict: bool = False):
    # --- STIMULUS -----------------------------------------------------------
    try:
        p = Params.from_list(p)
    except Exception:
        pass

    straight = stim
    delayed = func.synapse(stim, delay=p.delay * fs)

    output = np.multiply(straight, delayed)
    output = np.array(output) * p.nonlinearity_gain

    if return_dict:
        out = {"stim": stim, "an1": output, "p": p}
    else:
        if verbose_output:
            # out = (output, p)
            out = (straight, delayed, output, p)

        else:
            out = (output,)
    return out


if __name__ == "__main__":
    # from model_rebound_params_nonorm import Params
    p = Params()
    input_stim, s, ppau, pdur = utils.makePPFstim()
    out = run_model(input_stim, p)
    print(out)
