from ..neuralnodes import params
import dataclasses


@dataclasses.dataclass
class Params(params.Params):
    delay: float = 0.020
    nonlinearity_gain: float = 2.000000

    def get_bounds(self):
        param_dict = self.to_dict()
        bounds = []
        for key in param_dict.keys():
            if 'tau' in key or 'sigma' in key:
                bounds.append((0.000001, 0.020))
            elif 'gain' in key:
                bounds.append((0.0000001, 80))
            elif 'delay' in key:
                bounds.append((0.000001, 0.080))
            elif 'dur' in key:
                bounds.append((0.000001, 0.050))
            else:
                bounds.append((-10, 10))
        return bounds
