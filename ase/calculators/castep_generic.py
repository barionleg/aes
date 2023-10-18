"""CASTEP GenericFileIO calculator"""

import os
from pathlib import Path
from typing import Optional, Union
import warnings

from ase import Atoms
from ase.calculators.genericfileio import (
    GenericFileIOCalculator, CalculatorTemplate, read_stdout)
# from ase.io import write
from ase.io.castep import (read_castep_castep_new, write_cell_simple,
                           write_param_simple, read_bands)


################################
#  Castep Generic IO Template  #
################################

class CastepProfile:
    def __init__(self,
                 command: Optional[str] = None,
                 *,
                 pseudopotential_path: Union[Path, str, None] = None):

        if command:
            self.exe = command
        elif 'CASTEP_COMMAND' in os.environ:
            self.exe = os.environ['CASTEP_COMMAND']
        else:
            self.exe = 'castep.serial'

        self.pseudopotential_path = None
        if pseudopotential_path:
            self.pseudopotential_path = pseudopotential_path
        elif 'CASTEP_PP_PATH' in os.environ:
            self.pseudopotential_path = os.environ['CASTEP_PP_PATH']

    @staticmethod
    def parse_version(stdout):
        """Parse the version of castep from the executable"""
        import re
        match = re.match(r'CASTEP version: (\S+)', stdout, re.M)
        assert match is not None
        return match.group(1)

    def version(self):
        """Get the version of castep"""
        stdout = read_stdout(self.argv)
        return self.parse_version(stdout)

    def run(self, directory, seedname):
        """Define how to run Castep"""
        from subprocess import check_call
        argv = [self.exe, str(seedname)]
        print("running:", argv)

        if self.pseudopotential_path:
            run_env = os.environ.copy()
            run_env.update({'PSPOT_DIR': self})
        else:
            run_env = os.environ

        check_call(argv, cwd=directory, env=run_env)


class CastepTemplate(CalculatorTemplate):
    def __init__(self):
        """Initialise castep calculation definition"""
        super().__init__(
            name='castep',
            implemented_properties=['energy', 'free_energy', 'forces', 'stress'])
        self.seedname = 'castep'

    @staticmethod
    def _get_kpoint_params(atoms: Atoms,
                           parameters: dict) -> dict:
        """Get Castep .cell parameters from user 'kpts' specification"""
        if 'kpts' in parameters:
            return {'kpoint_mp_grid': parameters['kpts']}
        else:
            return {}

    def write_input(self, directory, atoms, parameters, properties):
        """Write the castep cell and param files"""

        from ase.io.castep import sort_atoms
        # Write the sorting/unsorting map so we can reorder atoms on read
        _ = sort_atoms(atoms,
                       sort_file=(directory / f'{self.seedname}.ase-sort.json'))

        # Separate parameters for seedname.cell and seedname.param files
        cell_params = parameters.get('cell', {}).copy()
        param_params = parameters.get('param', {}).copy()

        cell_params.update(self._get_kpoint_params(atoms, parameters))

        cellname = directory / (self.seedname + ".cell")
        with open(cellname, "w") as fd:
            write_cell_simple(fd, atoms, parameters=cell_params)

        if 'stress' in properties:
            param_params['CALCULATE_STRESS'] = 'true'

        paramname = directory / (self.seedname + ".param")
        with open(paramname, "w") as fd:
            write_param_simple(fd, parameters=param_params)

    def execute(self, directory, profile):
        """Execute castep"""
        profile.run(directory,
                    self.seedname)

    def read_results(self, directory):
        """Parse results from the .castep file and return them as a dict"""
        dotcastep_path = directory / (self.seedname + ".castep")
        with open(dotcastep_path) as fd:
            props = read_castep_castep_new(fd)

        dotbands_path = directory / (self.seedname + ".bands")
        with open(dotbands_path) as fd:
            kpts, weights, eigenvalues, efermi = read_bands(dotbands_path)
            props.update(kpts=kpts, weights=weights,
                         eigenvalues=eigenvalues, efermi=efermi)

        sortfile_path = directory / (self.seedname + ".ase-sort.json")
        if sortfile_path.is_file():
            from json import load as load_json
            with open(sortfile_path, 'r') as fd:
                mapping = load_json(fd)['castep_to_ase']

            props['forces'] = props['forces'][mapping]

        return props


################################
# The ASE calculator interface #
################################
class Castep(GenericFileIOCalculator):
    def __init__(self, *,
                 profile=None,
                 command=GenericFileIOCalculator._deprecated,
                 label=GenericFileIOCalculator._deprecated,
                 directory='.',
                 **kwargs):

        if label is not self._deprecated:
            warnings.warn('Ignoring label, please use directory instead',
                          FutureWarning)

        if command is not self._deprecated:
            raise RuntimeError(
                'Generic calculator does not use "command" argument, this '
                'should be passed to "profile" argument as CastepProfile')

        template = CastepTemplate()
        if profile is None:
            profile = CastepProfile()
        super().__init__(profile=profile, template=template,
                         directory=directory,
                         parameters=kwargs)
