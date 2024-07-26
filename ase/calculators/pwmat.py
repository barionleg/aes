"""Define a calculator for PWmat"""
import os
import re
from copy import deepcopy
from pathlib import Path

from ase.calculators.genericfileio import (BaseProfile, CalculatorTemplate,
                                           GenericFileIOCalculator)
from ase.io import read, write
from ase.io.pwmat_namelist.namelist import Namelist_pwmat
from ase.io.pwmat import write_IN_KPT


class PWmatProfile(BaseProfile):
    configvars = {'pseudo_dir'}

    def __init__(self, pseudo_dir, **kwargs):
        nprocs = os.environ.get('SLURM_NPROCS')
        if nprocs is None:
            raise Exception("It is necessary to submit the job \
to the computing cluster through \
the SLURM platform using sbatch command!")
        command = f'mpirun -np {nprocs} -iface ib0 PWmat | tee output'
        super().__init__(command, **kwargs)
        self.pseudo_dir = Path(pseudo_dir)

    def version(self):
        module_list = os.popen('module list').read()
        pattern = r'pwmat/(\d+\.\d+\.\d+)'
        ret = re.search(pattern, module_list)
        if ret:
            return ret.group(1)
        else:
            return 'pwmat version information not found'

    def get_calculator_command(self, inputfile):
        return []


class PWmatTemplate(CalculatorTemplate):
    def __init__(self):
        super().__init__(
            'pwmat',
            ['energy', 'free_energy', 'forces', 'stress', 'magmoms', 'dipole'],
        )
        self.inputname = 'etot.input'
        self.outputname = 'pwmat.out'
        self.errorname = 'ERRORS'

    def write_input(self, profile, directory, atoms, parameters, properties):
        """
            Write all inputfiles, including etot.input and atom.config.
        Args:
            profile: PWmatProfile

            directory: str
                Set the working directory.

            atoms: Atoms
                Attach an atoms object to the calculator.

            parameters: dict
                Parameters for etot.input and atom.config.
        """
        dst = directory / self.inputname
        pp_dir = Path(os.path.abspath(profile.pseudo_dir))
        input_data = Namelist_pwmat(parameters.pop('input_data', None))
        input_data.to_nested()
        if 'IN.PSP' in list(input_data):
            if not isinstance(input_data['IN.PSP'], list):
                raise ValueError("The keyword IN.PSP must be List")
            in_psp_tmp = deepcopy(input_data['IN.PSP'])
            for i, psp in enumerate(in_psp_tmp):
                pp_path = pp_dir / psp
                if not os.path.exists(pp_path):
                    raise FileNotFoundError("Pseudopotential file \
does not exist.")
                else:
                    input_data['IN.PSP'][i] = pp_path
        else:
            raise KeyError('The keyword IN.PSP must be given !')
        parameters['input_data'] = input_data

        write(
            dst,
            atoms,
            format='pwmat-in',
            properties=properties,
            **parameters,
        )

        config_name = parameters['input_data'].get('IN.ATOM', None)
        if config_name is None:
            config_name = 'atom.config'
        dst_config = directory / config_name
        write(dst_config, atoms, format='pwmat', **parameters)

        IN_KPT = parameters['input_data'].get('IN.KPT', None)
        if IN_KPT == 'T' or IN_KPT is True:
            dst_kpt = directory / 'IN.KPT'
            write_IN_KPT(dst_kpt, atoms, **parameters)

    def execute(self, directory, profile):
        profile.run(directory, self.inputname, self.outputname,
                    errorfile=self.errorname)

    def read_results(self, directory):
        """
        Potential energy and fermi energy are read from REPORT.
        Positions, cells, stresses and forces are read from MOVEMENT
        when this file exists.
        The magnetic values for each atom are read from OUT.QDIV, which will
        appear when CHARGE_DECOMP is set to T in etot.input.

        """
        path = directory / 'REPORT'
        path_movement = directory / 'MOVEMENT'
        path_qdiv = directory / 'OUT.QDIV'
        atoms = read(path, format='pwmat-report',
                     MOVEMENT=path_movement, QDIV=path_qdiv, index=-1)
        return dict(atoms.calc.properties())

    def load_profile(self, cfg, **kwargs):
        return PWmatProfile.from_config(cfg, self.name, **kwargs)

    def socketio_parameters(self, unixsocket, port):
        return {}

    def socketio_argv(self, profile, unixsocket, port):
        if unixsocket:
            ipi_arg = f'{unixsocket}:UNIX'
        else:
            ipi_arg = f'localhost:{port:d}'  # XXX should take host, too
        return profile.get_calculator_command(self.inputname) + [
            '--ipi',
            ipi_arg,
        ]


class PWmat(GenericFileIOCalculator):
    def __init__(
        self,
        *,
        profile=None,
        directory='.',
        **kwargs,
    ):
        """
        Set up PWmat calculator with input parameters.

        Args:
            input_data: dict
                A dictionary with input parameters for PWmat.
            kspacing: float
                Generate a grid of k-points with this as the minimum distance
                in A^-1 between them in reciprocal space.
            sort: bool
                Sort the atomic indices alphabetically by element,
                default to True.
            ignore_constraints: bool
                Ignore all constraints on `atoms`, default to False.
            show_magnetic: bool
                Whether to write MAGNETIC label in atom.config,
                default to False.
            density: float
                 Number of k-points per 1/A on the output kpts list.
                 default to None.
                 Being used to generate the k-points list for IN.KPT file.

        - Examples:
        >>> from ase.calculators.pwmat import PWmat, PWmatProfile
        >>> from ase.build import bulk
        >>> atoms = bulk('Fe',crystalstructure='fcc',a=3.43,cubic=True)
        >>> input_data = {
            'Parallel':[1,4],
            'JOB':'SCF',
            'IN.ATOM':'atom.config',
            'IN.PSP':['Fe-sp.PD04.PBE.UPF'],
            'XCFUNCTIONAL':'PBE',
            }
        >>> profile = PWmatProfile(pseudo_dir='.')
        >>> calc = PWmat(profile=profile,input_data=input_data,kspacing=0.04)
        >>> atoms.calc = calc
        >>> atoms.get_potential_energy()

        """

        super().__init__(
            template=PWmatTemplate(),
            profile=profile,
            directory=directory,
            parameters=kwargs,
        )
