import sys
import numpy as np
from ase.calculators.calculator import all_changes
from ase.calculators.calculator import FileIOCalculator
from ase.calculators.genericfileio import GenericFileIOCalculator
import os
import shutil
from ase.io.n2p2 import read_n2p2, write_n2p2
from tempfile import mkdtemp#, NamedTemporaryFile, mktemp as uns_mktemp
from ase import units 

class N2P2Template:
    '''example: 
        N2P2Calculator( 
            directory = 'tmp',
            files = [
                'input.nn',
                'scaling.data',
                'weights.008.data',
                'weights.001.data'],
            )
    '''
    
    command = 'nnp-predict 0'
    'Command used to start calculation'

    name = 'n2p2'

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                    label=None, atoms=None, command=None, files=[],
                    model_length_units = units.Ang, #model units to ASE units, i.e. units.Bohr
                    model_energy_units = units.eV,
                    **kwargs):
        """File-IO calculator.

        command: str
            Command used to start calculation.
        """

        self.files = files

        #FileIOCalculator.__init__(self, restart, ignore_bad_restart_file, label,
        #                    atoms, **kwargs)
        GenericFileIOCalculator.__init__(self, 
            #restart, ignore_bad_restart_file, label,
            #                atoms, 
                            **kwargs)

        if command is not None:
            self.command = command
        else:
            name = 'ASE_' + self.name.upper() + '_COMMAND'
            self.command = os.environ.get(name, self.command)
            
        self.implemented_properties = {
            'energy' : self.calculate,
            'forces' : self.calculate}
        self.results = {}

        ## preparing
        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        self.write_files()

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically."""

        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)
            
        write_n2p2(
            os.path.join(self.directory, 'input.data'),
            atoms,
            with_energy_and_forces = False)

    def write_files(self): #should this be initialize?
        for filename in self.files:
            src = filename
            basename = os.path.basename(filename)
            dest = os.path.join(self.directory, basename)
            shutil.copyfile(src, dest)

    def read_results(self):
        res_atoms = read_n2p2(
                    filename= os.path.join(self.directory,'output.data'),
                    index=-1, 
                    with_energy_and_forces = True)
        self.results = res_atoms.calc.results




class N2P2GenericFileIOCalculator(GenericFileIOCalculator):
    '''example: 
        N2P2Calculator( 
            directory = 'tmp',
            files = [
                'input.nn',
                'scaling.data',
                'weights.008.data',
                'weights.001.data'],
            )
    '''
    
    command = 'nnp-predict 0'
    'Command used to start calculation'

    name = 'n2p2'

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, command=None, files=[], **kwargs):
        """File-IO calculator.

        command: str
            Command used to start calculation.
        """

        self.files = files

        GenericFileIOCalculator.__init__(self, 
            #restart, ignore_bad_restart_file, label,
            #                atoms, 
                            **kwargs)

        if command is not None:
            self.command = command
        else:
            name = 'ASE_' + self.name.upper() + '_COMMAND'
            self.command = os.environ.get(name, self.command)
            
        self.implemented_properties = {
            'energy' : self.calculate,
            'forces' : self.calculate}
        self.results = {}

        ## preparing
        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        self.write_files()

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically."""

        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)
            
        write_n2p2(
            os.path.join(self.directory, 'input.data'),
            atoms,
            with_energy_and_forces = False)

    def write_files(self): #should this be initialize?
        for filename in self.files:
            src = filename
            basename = os.path.basename(filename)
            dest = os.path.join(self.directory, basename)
            shutil.copyfile(src, dest)

    def read_results(self):

        res_atoms = read_n2p2(
                    filename= os.path.join(self.directory,'output.data'),
                    index=-1, 
                    with_energy_and_forces = True)

        self.results = res_atoms.calc.results




from contextlib import contextmanager
from pathlib import Path
import subprocess

class N2P2Calculator(FileIOCalculator):
    '''example: 
        N2P2Calculator( 
            directory = 'tmp',
            files = [
                'input.nn',
                'scaling.data',
                'weights.008.data',
                'weights.001.data'],
            )
    '''
    
    command = 'nnp-predict 0'
    'Command used to start calculation'

    name = 'n2p2'

    def __init__(self, restart=None, 
                label=None, atoms=None, command=None,
                files=[], txt='n2p2.out', keep_tmp_files=False,
                model_length_units = units.Ang, #model units to ASE units, i.e. units.Bohr
                model_energy_units = units.eV,
                **kwargs):
        """File-IO calculator.

        command: str
            Command used to start calculation.
        """

        self.files = files
        self.model_length_units = model_length_units
        self.model_energy_units = model_energy_units
        FileIOCalculator.__init__(self, restart, **kwargs)

        if command is not None:
            self.command = command
        else:
            name = 'ASE_' + self.name.upper() + '_COMMAND'
            self.command = os.environ.get(name, self.command)
            
        self.implemented_properties = {
            'free_energy': self.calculate,
            'energy' : self.calculate,
            'forces' : self.calculate}
        self.results = {}
        
        self.txt=txt
        self.keep_tmp_files = keep_tmp_files

        ## preparing
        if self.directory is None:
            self.directly = mkdtemp(prefix="tmp-")
        if self.directory != os.curdir: #:and not os.path.isdir(self.directory):
            os.makedirs(self.directory, exist_ok=True)
        else:
            # lets not delete the initial folder so we don't delete a users script
            self.keep_tmp_files  = True
            
        self.write_files()

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically."""

        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)
            
        atoms_model_units = atoms.copy()
        atoms_model_units.set_positions(atoms.get_positions()/self.model_length_units)
        atoms_model_units.set_cell(atoms.get_cell()/self.model_length_units)
        
        write_n2p2(
            os.path.join(self.directory, 'input.data'),
            atoms_model_units,
            with_energy_and_forces = False)

    def write_files(self): #should this be initialize?
        for filename in self.files:
            src = filename
            basename = os.path.basename(filename)
            dest = os.path.join(self.directory, basename)
            shutil.copyfile(src, dest)



    @contextmanager
    def _txt_outstream(self):
        ## copied from the vasp calculator and lightly changed
        """Custom function for opening a text output stream. Uses self.txt
        to determine the output stream, and accepts a string or an open
        writable object.
        If a string is used, a new stream is opened, and automatically closes
        the new stream again when exiting.

        Examples:
        # Pass a string
        calc.txt = 'vasp.out'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # Redirects the stdout to 'vasp.out'

        # Use an existing stream
        mystream = open('vasp.out', 'w')
        calc.txt = mystream
        with calc.txt_outstream() as out:
            calc.run(out=out)
        mystream.close()

        # Print to stdout
        calc.txt = '-'
        with calc.txt_outstream() as out:
            calc.run(out=out)   # output is written to stdout
        """

        txt = self.txt
        open_and_close = False  # Do we open the file?

        if txt is None:
            # Suppress stdout
            out = subprocess.DEVNULL
        else:
            if isinstance(txt, str):
                if txt == '-':
                    # subprocess.call redirects this to stdout
                    out = None
                else:
                    # Open the file in the work directory
                    txt = Path(self.directory) / txt
                    # We wait with opening the file, until we are inside the
                    # try/finally
                    open_and_close = True
            elif hasattr(txt, 'write'):
                out = txt
            else:
                raise RuntimeError('txt should either be a string'
                                   'or an I/O stream, got {}'.format(txt))

        try:
            if open_and_close:
                out = open(txt, 'w')
            yield out
        finally:
            if open_and_close:
                out.close()

    def execute(self): 
        # lightly modified from the original fileIO form to dump the stderr and  
        # stdout to a log file since there can be so many extrapolation warnings
        if self.command is None:
            raise CalculatorSetupError(
                'Please set ${} environment variable '
                .format('ASE_' + self.name.upper() + '_COMMAND') +
                'or supply the command keyword')
        command = self.command
        if 'PREFIX' in command:
            command = command.replace('PREFIX', self.prefix)

        try:
            with self._txt_outstream() as out:
                proc = subprocess.Popen(command, 
                                        shell=True,                                    
                                        stdout=out,
                                        stderr=out,
                                        cwd=self.directory)
        except OSError as err:
            # Actually this may never happen with shell=True, since
            # probably the shell launches successfully.  But we soon want
            # to allow calling the subprocess directly, and then this
            # distinction (failed to launch vs failed to run) is useful.
            msg = 'Failed to execute "{}"'.format(command)
            raise EnvironmentError(msg) from err

        errorcode = proc.wait()

        if errorcode:
            path = os.path.abspath(self.directory)
            msg = ('Calculator "{}" failed with command "{}" failed in '
                   '{} with error code {}'.format(self.name, command,
                                                  path, errorcode))
            raise CalculationFailed(msg)


    def read_results(self):
        res_atoms = read_n2p2(
                    filename= os.path.join(self.directory,'output.data'),
                    index=-1, 
                    with_energy_and_forces = True,
                    model_length_units = self.model_length_units,
                    model_energy_units = self.model_energy_units)
        self.results = res_atoms.calc.results
        
    def clean(self):
        if not self.keep_tmp_files:
            shutil.rmtree(self.directory)





def parse_input_file(input_file): 
    
    fid=open(input_file,'r')
    lines_with_comments=fid.readlines()
    fid.close()
    
    parameters = {}
    symfunctions = []
    atom_energy = {}
    for line in lines_with_comments:
        if line[0] != '#':
            comment_free_line = line.split('#')[0]
            line_parts = comment_free_line.split()
            if len(line_parts)>0:# skip empty
                if line_parts[0] == 'symfunction_short': #the symfunction keywords are probably the only ones that repeat.
                    symfunctions.append(line_parts)
                elif line_parts[0] == 'atom_energy':
                    atom_energy[line_parts[1]]=float(line_parts[2])
                else:
                    if len(line_parts)==1:
                        line_parts.append(None) # some parameters are on-off flags so we need a value to store, maybe the value should be '' to make writting easier
                    parameters[line_parts[0]]=line_parts[1:]
                    ### we could do this so that we don't create single element lists
                    #if len(line_parts)>2:
                    #    parameters[line_parts[0]]=line_parts[1:]
                    #else:   
                    #    parameters[line_parts[0]]=line_parts[1]
    parameters['atom_energy'] = atom_energy
    return parameters, symfunctions


def write_input_file(filename, parameters, symfunctions):
    
    fid = open(filename,'w')
    
    for key in parameters.keys():
        if key=='atom_energy':
            atom_energy = parameters['atom_energy']
            for element in atom_energy.keys():
                fid.write('atom_energy   %s   %f\n'%(element, atom_energy[element]))
        else:
            fid.write(str(key))
            for value in parameters[key]:
                fid.write(' '+str(value))
            fid.write('\n')
        
    for symfunc in symfunctions:
        for value in symfunc:
            fid.write(value + ' ')
        fid.write('\n')
    fid.close() 
    
    
    
    
    

def radial_grid(N_r, r_0, r_c, width_scale, mode):
    N        = N_r
    grid, dr = np.linspace(r_0, r_c, N, endpoint=False, retstep=True)
    r_N = grid[-1]
    if mode == "center":
        eta_grid = 0.5 / (grid * width_scale)**2
        rs_grid = [0.0] * N
    elif mode == "shift":
        eta_grid = [0.5 / (dr * width_scale)**2 ] * N
        rs_grid = grid
    return rs_grid, eta_grid


def generate_symfunctions_G2(elements, radkwrds, annotate = True):
    rs_grid, eta_grid = radial_grid(**radkwrds)
    r_c = radkwrds['r_c']
    symfunctions = []
    
    if annotate:
        symfunctions.append([''])
        symfunctions.append(["# Generating radial symmetry function set:"  ])
        symfunctions.append(["# mode  = {0:9s}".format(radkwrds['mode'])   ])
        symfunctions.append(["# r_0   = {0:9.3E}".format(radkwrds['r_0'])  ])
        symfunctions.append(["# r_c   = {0:9.3E}".format(radkwrds['r_c'])  ])
        #symfunctions.append(["# r_N   = {0:9.3E}".format(r_N)])
        symfunctions.append(["# N     = {0:9d}".format(radkwrds['N_r'])    ])
        symfunctions.append(["# r_s  = " + " ".join(str(_) for _ in rs_grid)   ])
        symfunctions.append(["# eta  = " + " ".join(str(_) for _ in eta_grid)  ])
        
    for e0 in elements:
        if annotate:
            symfunctions.append(["# Radial symmetry functions for element {0:2s}".format(e0)])
        for e1 in elements:
            for (eta, rs) in zip(eta_grid, rs_grid):
                line = "symfunction_short {0:2s} 2 {1:2s} {2:9.3E} {3:9.3E} {4:9.3E}".format(e0, e1, eta, rs, r_c)
                symfunctions.append(line.split())
    return symfunctions

def generate_symfunctions_G9(elements, radkwrds, zetas, annotate = True):
    rs_grid, eta_grid = radial_grid(**radkwrds)
    r_c = radkwrds['r_c']
    symfunctions = []

    if annotate:
        symfunctions.append([''])
        symfunctions.append(["# Generating wide angular symmetry function set:"])
        symfunctions.append(["# mode  = {0:9s}".format(radkwrds['mode'])       ])
        symfunctions.append(["# r_0   = {0:9.3E}".format(radkwrds['r_0'])      ])
        symfunctions.append(["# r_c   = {0:9.3E}".format(radkwrds['r_c'])      ])
        #symfunctions.append(["# r_N   = {0:9.3E}".format(r_N)])
        symfunctions.append(["# N     = {0:9d}".format(radkwrds['N_r'])        ])
        symfunctions.append(["# r_s  = " + " ".join(str(_) for _ in rs_grid)   ])
        symfunctions.append(["# eta  = " + " ".join(str(_) for _ in eta_grid)  ])
        symfunctions.append(["# zetas = " + " ".join(str(z) for z in zetas)    ])
        
    
    for e0 in elements:
        if annotate:
            symfunctions.append(["# Wide angular symmetry functions for element {0:2s}".format(e0)])
        for e1 in elements:
            elements_reduced = elements[elements.index(e1):]
            for e2 in elements_reduced:
                for (eta, rs) in zip(eta_grid, rs_grid):
                    for zeta in zetas:
                        for lambd in [-1.0, 1.0]:
                            line = "symfunction_short {0:2s} 9 {1:2s} {2:2s} {3:9.3E} {4:2.0f} {5:9.3E} {6:9.3E} {7:9.3E}".format(e0, e1, e2, eta, lambd, zeta, r_c, rs)
                            symfunctions.append(line.split())
    return symfunctions
    
    


from ase.calculators.test import numeric_stress

def AttachNumericStresses(SubCalculator, calckwrds, d=1e-6, voigt=True):

    class NumericStressWrapper(SubCalculator):
        def __init__(self, SubCalculator, calckwrds, d=d, voigt=voigt):
            #self.calc = SubCalculator(**calckwrds)
            
            super().__init__(**calckwrds)
            self.implemented_properties.append('stress')
            self.implemented_properties.append('stresses')
            ## these long names should avoid any attribute conflicts
            self.numeric_stress_d = d
            self.numeric_stress_voigt = voigt
            
        def calculate(self, atoms=None, properties=['energy','forces'], system_changes=all_changes):
            """ Calculates all the specific property for each calculator and returns with the summed value. 
            """
            
            #Calculator.calculate(self, atoms, properties, system_changes) # do we need this for a simple wrapper?
            
            properties_without_stress = []
            for prop in properties:
                if prop not in ['stress', 'stresses']:
                    properties_without_stress.append(prop)

            if 'stress' in properties or 'stresses' in properties:
                # The point here is to avoid doing 6 calculations unless explicity asked. 
                # if you ask for energy/forces, they'll be calculated but not stresses.
                # if you ask for stresses, the energy/forces will also be calulated since 
                # those are required for the numerical stress calculation
                # there might be 1 or 2 calculator calls than can be avoided by reordering, 
                # but numeric stresses already require 12 calls, so it's not a large savings 
                atoms_with_cell_fd = atoms.copy()
                atoms_with_cell_fd.calc = super()
                stress = numeric_stress(atoms_with_cell_fd, 
                    d=self.numeric_stress_d,
                    voigt=self.numeric_stress_voigt)
            
            super().calculate(atoms, properties_without_stress, system_changes)

            if 'stress' in properties or 'stresses' in properties:
                self.results['stress'] = stress
                self.results['stresses'] = stress

    return NumericStressWrapper(SubCalculator,calckwrds)
