import neuralnodes.params
import dataclasses


@dataclasses.dataclass
class Params(neuralnodes.params.Params):
    filter_inh_dur: float = 8.0
    filter_inh_sigma: float = 2.0
    filter_inh_gain: float = 1.0

    filter_delay: float = 7.0

    filter_exc_dur: float = 4.0
    filter_exc_sigma: float = 0.5
    filter_exc_gain: float = 0.1

    nonlinearity_slope: float = 1.500000
    nonlinearity_shift: float = 1.500000
    nonlinearity_gain: float = 5.000000
    nonlinearity_baseline: float = -0.500000


    def _make_bounds(self):
        self.bounds = {
            'filter_inh_dur': (0, 50),
            'filter_inh_sigma': (0, 10),
            'filter_inh_gain': (0, 10),
            'filter_delay': (0, 200),
            'filter_exc_dur': (0, 50),
            'filter_exc_sigma': (0, 10),
            'filter_exc_gain': (0, 10),
            'nonlinearity_slope': (0, 100),
            'nonlinearity_shift': (-100, 100),
            'nonlinearity_gain': (0, 100),
            'nonlinearity_baseline': (-100, 100),
        }
        return self.bounds

    def get_bounds(self):
        self._make_bounds()
        return list(self.bounds.values())
