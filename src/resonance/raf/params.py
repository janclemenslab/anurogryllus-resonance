from ..neuralnodes.params import Params
import dataclasses
from typing import Tuple


class Param():
    name: str
    value: float
    bounds: Tuple[float]


@dataclasses.dataclass
class Params(Params):
    stim_gain: float = 1.0
    damping: float = -5.0
    frequency: float = 10.0
    output_gain: float = 2.0
    # tau_x: float = 1.0
    # tau_y: float = 1.0
    # nonlinearity_slope: float = 1.5
    # nonlinearity_shift: float = 1.5
    # nonlinearity_gain: float = 2.0
    # nonlinearity_baseline: float = 0.0

    def _make_bounds(self):
        param_dict = self.to_dict()
        self.bounds = dict()
        for key in param_dict.keys():
            if 'tau' in key or 'sigma' in key:
                self.bounds[key] = (0.0, 40.0)
            elif 'delay' in key:
                self.bounds[key] = (0.0, 80.0)
            elif 'dur' in key:
                self.bounds[key] = (1.0, 50.0)
            elif 'damping' in key:
                self.bounds[key] = (-1.0, -0.0)
            elif 'frequency' in key:
                self.bounds[key] = (10, 500)
            elif 'gain' in key:
                self.bounds[key] = (0, 10)
            else:
                self.bounds[key] = (-10, 10)
        return self.bounds

    def get_bounds(self):
        self._make_bounds()
        return list(self.bounds.values())
