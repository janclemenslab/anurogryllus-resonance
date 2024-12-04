import dataclasses
import yaml
import collections


@dataclasses.dataclass
class Params():

    def get_bounds(self):
        param_dict = self.to_dict()
        bounds = []
        for key in param_dict.keys():
            if 'tau' in key or 'sigma' in key:
                bounds.append((0, 10_000))
            elif 'delay' in key:
                bounds.append((0, 21))
            elif 'dur' in key:
                bounds.append((2, 10_000))
            else:
                bounds.append((None, None))
        return bounds

    def to_yaml(self, filename):
        with open(filename, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, filename: str):
        with open(filename, 'r') as f:
            return cls(**yaml.load(f, Loader=yaml.SafeLoader))

    def to_m(self, filename: str):
        """Save as matlab file."""

        with open(filename, "w") as f:
            for key, val in self.to_dict().items():
                print(f"p.{key} = {val:0.4f};", file=f)

    def to_dict(self):
        dct = dataclasses.asdict(self)
        # cast all to floats so yaml looks proper
        for key, val in dct.items():
            dct[key] = float(val)
        return dct

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_list(self):
        return list(self.to_dict().values())

    @classmethod
    def from_list(cls, list):
        return cls(*list)

    def to_namedtuple(self):
        params_dict = self.to_dict()
        params = collections.namedtuple('Params', params_dict.keys())
        p = params(**params_dict)
        return p
    # @classmethod
    # def from_torch_tensor(cls, torch_array):
    # 	# TODO
    # 	raise NotImplementedError

    # def to_torch_tensor(self):
    # 	return torch.tensor(self.to_list())
