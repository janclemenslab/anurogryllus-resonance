from ..neuralnodes import params
import dataclasses


@dataclasses.dataclass
class Params(params.Params):
    filter_inh_dur: float = 8.0
    filter_inh_gain: float = 1.0
    filter_delay: float = 7.0
    filter_exc_dur: float = 4.0
    filter_exc_gain: float = 0.1
    ln2_filter_exc_dur: float = 1
    ln2_filter_exc_gain: float = 1
    ln2_filter_inh_dur: float = 1
    ln2_filter_inh_gain: float = 1
    ln2_ln4_delay: float = 1
    ln2_ln4_gain:float =  -1



    def _make_bounds(self):
        self.bounds = {
            'filter_inh_dur': (0, 100),
            'filter_inh_gain': (0, 10),
            'filter_delay': (0, 200),
            'filter_exc_dur': (0, 100),
            'filter_exc_gain': (0, 10),
        }
        return self.bounds

    def get_bounds(self):
        self._make_bounds()
        return list(self.bounds.values())
