import os
import warnings
from ase.io.pwmat_namelist.keys import pwmat_keys
from collections import UserDict
from collections.abc import MutableMapping


class Namelist_pwmat(UserDict):
    """
    See :func: 'ase.io.espresso_namelist.namelist' for details.
    """
    def __getitem__(self, key):
        return super().__getitem__(key.upper())

    def __setitem__(self, key, value):
        super().__setitem__(
            key.upper(), Namelist_pwmat(value) if isinstance(
                value, MutableMapping) else value)

    def __delitem__(self, key):
        super().__delitem__(key.upper())

    def to_string(self, list_form: bool = False):
        etot_input = []
        for key, value in self.items():
            if value is True or value == 'T':
                etot_input.append(f'{key} = T\n')
            elif value is False or value == 'F':
                etot_input.append(f'{key} = F\n')
            elif key == 'PARALLEL':
                etot_input.append('{}   {}\n'.format(*value))
            elif key != 'PARALLEL' and isinstance(value, list):
                for n, v in enumerate(value):
                    etot_input.append(f'{key}{n+1} = {v}\n')
            else:
                etot_input.append(f'{key} = {value}\n')
        if list_form:
            return etot_input
        else:
            return "".join(etot_input)

    def to_nested(self, warn: bool = False, sorted_keys: bool = False,
                  **kwargs):
        keys = pwmat_keys
        unused_keys = []
        constructed_namelist = {}
        nprocs = os.environ.get('SLURM_NPROCS')

        if 'PARALLEL' not in list(self):
            constructed_namelist['PARALLEL'] = [1, nprocs]
        else:
            assert len(self['PARALLEL']) == 2
            tmp_list = [int(n) for n in self['PARALLEL']]
            if nprocs is not None:
                assert tmp_list[0] * tmp_list[1] == int(nprocs)

        for arg_key in list(self):
            if arg_key in keys:
                value = self.pop(arg_key)
                constructed_namelist[arg_key] = value
            else:
                self.pop(arg_key)
                unused_keys.append(arg_key)
        for arg_key in list(kwargs):
            if arg_key in keys:
                value = kwargs.pop(arg_key)
                constructed_namelist[arg_key] = value
            else:
                unused_keys.append(arg_key)
        if unused_keys and warn:
            warnings.warn(f"Unused keys: {', '.join(unused_keys)}")

        if sorted_keys:
            constructed_namelist = dict(sorted(constructed_namelist.items(),
                                               key=lambda item:
                                                   keys.index(item[0])))
        super().update(Namelist_pwmat(constructed_namelist))
