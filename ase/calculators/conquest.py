"""This module defines an ASE interface to CONQUEST.

http://www.order-n.org/
Authors: J Kane Shenton <jkshenton@gmail.com>
         Jack Baker
         J Poulton
         Zamaan Raza
         L Truflandier <lionel.truflandier@u-bordeaux.fr>
"""
from ase.units import Hartree
from pathlib import Path
from shutil import copy
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
from ase.io.conquest import ConquestEnv, conquest_err, conquest_warn


class Conquest(FileIOCalculator):

    implemented_properties = ['energy', 'forces', 'stress']
    command = None
    conquest_infile = "Conquest_input"
    conquest_outfile = "Conquest_out_ase"

    # A minimal set of defaults
    default_parameters = {
        'grid_cutoff': 100,
        'kpts': None,
        'xc': 'PBE',
        'self_consistent': True,
        'scf_tolerance': 1.0e-6,
        'nspin': 1,
        'general.pseudopotentialtype': 'Hamann',
        'io.iprint': 1,
        'io.fractionalatomiccoords': True,
        'sc.maxiters': 50,
        'diag.smearingtype': 1,
        'diag.kt': 0.001,
        'atommove.typeofrun': 'static',
        'dm.solutionmethod': 'diagon',
        'io.writeouttoasefile': True}

    def __init__(self, restart=None, label=None, atoms=None, basis={},
                 **kwargs):
        """
        ASE-Calculator class for Conquest.

        Parameters
        ==========
        atoms: ASE atoms object
            (mandatory) an atoms object constructed either via ASE or
        read from an input.
        basis: dict
            (mandatory) a dictionary specifying the pseudopotential/basis
            files.
        directory: str
            directory used for storing input/output and calculation files
            (default None).
        label: str
            basename for working files (only used by ASE, eg. NEB)
            (default None)
        kpts: list or tuple
            k-points grid (default None).
        grid_cutoff: float
            integration grid in Ha (default 100).
        xc: str
            exchange and correlation functional (default 'PBE').
        self_consistent: bool
            choose either a SCF or non-SCF (default True).
        scf_tolerance: float
            Self-consistent-field convergence tolerance in Ha (default 1.0e-6).
        nspin: int
            Spin polarisation: 1 for unpolarized (default) ; 2 for polarised.
        conquest_flags: dict
            Other Conquest keyword arguments
        """

        self.initialised = False
        self.label = label
        self.eigenvalues = None
        self.occupancies = None
        self.kpoints = None
        self.weights = None
        self.nspin = None
        self.coorfile = None

        FileIOCalculator.__init__(self, restart=restart, label=self.label,
                                  atoms=atoms, **kwargs)

        self.basis = {}
        for spec in basis:
            self.basis[spec] = \
                Parameters({k.lower(): basis[spec][k] for k in basis[spec]})
            print('input basis:')
            print(spec, self.basis[spec])

        self.parameters = Parameters({})

        for key in kwargs:
            self.parameters[key.lower()] = kwargs[key]

        for key in self.default_parameters:
            if key not in self.parameters:
                self.parameters[key] = self.default_parameters[key]

    def init_basis(self, atoms, basis, ion_dir):
        """
        Determine whether to make basis sets useing MakeIonFiles (part of
        CONQUEST distribution). Parse .ion files to write basis information for
        Conquest_input. Make .ion files if necessary.
        """

        from ase.io.conquest import make_ion_files, sort_by_atomic_number

        # Get the species list
        self.species_list = sort_by_atomic_number(atoms)

        # Get XC for ion generation as input XC ; use PBE if not
        for key in self.parameters:
            if (key == 'xc'):
                ion_xc = self.parameters[key]
                break
            else:
                ion_xc = 'PBE'

        gen_basis = []
        for species in self.species_list:
            if ("gen_basis" in basis[species]):
                if (basis[species]["gen_basis"]):
                    gen_basis.append(True)
                else:
                    gen_basis.append(False)
            else:
                basis[species]["gen_basis"] = False
                gen_basis.append(False)

        count = 0
        # Test weither or not if we need to generate the basis
        for test in gen_basis:
            count += 1

            # TODO rewrite make_ion_files to avoid what is below
            basis_single = {k: basis[k] for k in self.species_list}
            species_single_list = [self.species_list[count - 1]]
            species_single = self.species_list[count - 1]

            # Yes: run make_ion_files
            if (test):
                if ('xc' in basis_single[species_single]):

                    if (ion_xc != basis_single[species_single]['xc']):
                        ion_xc = basis_single[species_single]['xc']
                        conquest_warn("{} xc {} enforced for basis \
                                      generation".format(species_single,
                                      ion_xc))

                        # basis_single[species_single]['xc'] = ion_xc

                print('\nmake_ion_files input:')
                print(species_single, basis_single[species_single])
                make_ion_files(basis_single, species_single_list,
                               directory=self.directory, xc=ion_xc)

            # No: try to find the ion file
            else:
                self.find_ion_file(species_single, basis, ion_dir, ion_xc)

        self.initialised = True

    # Try to locate .ion files
    def find_ion_file(self, species, basis, ion_dir, ion_xc):

        # Default name used for the input
        dname = species + ".ion"

        # Name of the file from basis dictionary
        if ("file" in basis[species]):
            fname = basis[species]["file"]

        # Default otherwise
        else:
            fname = dname

        if ("directory" in basis[species]):
            dname_ = basis[species]["directory"] + '/'
        else:
            dname_ = '.'

        # Default working directory
        fullpath = Path(self.directory).joinpath(dname)

        if (not ion_dir or ion_dir == '.' or ion_dir == './'
                or ion_dir == '/'):
            ion_dir = Path('')
        else:
            ion_dir = Path(ion_dir)

        if (species not in basis):
            basis[species] = {}

        ion_file_path_ = ion_dir.joinpath(Path(dname_ + fname))
        ion_file_path_lib = ion_dir.joinpath(Path("lib/" + fname))
        ion_file_path = ion_dir.joinpath(Path(fname))

        if ("xc" in basis[species]):
            ion_file_path_xc = ion_dir.joinpath(Path(basis[species]["xc"]
                                                     + "/" + species + '/'
                                                     + fname))
        else:
            ion_file_path_xc = ion_dir.joinpath(ion_xc + "/" + species + '/'
                                                + fname)

        count = 0
        print()
        for ion_file in [ion_file_path_,
                         ion_file_path,
                         ion_file_path_lib,
                         ion_file_path_xc,
                         Path(fname),
                         Path(self.directory).joinpath(fname)]:

            print("Try to find {} in {}".format(fname, ion_file), end="")
            if (ion_file.is_file() and ion_file != fullpath):
                print("... Found")
                copy(ion_file, fullpath)
                conquest_warn("{} copied into {}".format(ion_file, fullpath))
                count += 1
                break

            elif (ion_file.is_file() and ion_file == fullpath):
                print("... Found")
                conquest_warn("{} copied into {}".format(ion_file, fullpath))
                count += 1
                break

            else:
                print("... Not found")

        if (count == 0):
            raise conquest_err("Ion file {} not found".format(fname))

    def write_input(self, atoms, properties=None, system_changes=None):
        """
        Create the directory, make the basis if necessary, write the
        Conquest_input and coordinate files
        """

        from ase.io.conquest import write_conquest_input, write_conquest

        # set io.title
        if ('io.title' not in self.parameters):
            self.parameters['io.title'] = atoms.get_chemical_formula()

        # set coordfile from io.title
        if ('io.coordinates' not in self.parameters):
            coordfile = self.parameters['io.title'] + '.in'
            self.parameters['io.coordinates'] = coordfile
        else:
            coordfile = self.parameters['io.coordinates']

        # super().write_input(atoms, properties, system_changes)

        if not self.initialised:  # make the basis once only
            cq_env = ConquestEnv()
            try:
                pseudo_dir = cq_env.get('pseudo_path')
            except pseudo_dir.DoesNotExist:
                pseudo_dir = './'
            self.init_basis(atoms, self.basis, ion_dir=pseudo_dir)

        cq_in = Path(self.directory).joinpath(Path(self.conquest_infile))
        cq_coord = Path(self.directory).joinpath(Path(coordfile))

        with cq_in.open(mode='w') as infile_obj:

            write_conquest_input(infile_obj, atoms, self.species_list,
                                 self.parameters,
                                 directory=self.directory, basis=self.basis)

        with cq_coord.open(mode='w') as coordfile_obj:
            write_conquest(coordfile_obj, atoms, self.species_list)

        # it looks label is needed when using some ASE routines, eg. NEB
        if self.label:
            self.parameters.write(self.label + '.ase')

    def read_results(self, atoms=None):
        """
        Parse energy, Fermi level, forces, stresses from Conquest_out
        """

        from ase.io.conquest import read_conquest_out

        if (atoms is not None):
            self.atoms = atoms

        cq_out_path = Path(self.directory).joinpath(Path(self.conquest_outfile))

        if (not cq_out_path.is_file()):
            raise ReadError

        with cq_out_path.open(mode='r') as outfile_obj:
            self.results = read_conquest_out(outfile_obj, self.atoms)

        if (self.parameters['dm.solutionmethod'] == 'diagon'):
            if (self.parameters['io.iprint'] >= 1):
                self.get_bands(cq_out_path)

    def get_bands(self, cq_out_path):
        """
        Parser fermi level, eigenvalues, kpoints and weights from Conquest_out.
        Generates np.array(nspin, nkpoints, nbands) for eigenvalues and
        occupancies, np.array(3, nkpts) and np.array(nkpts) for kpoints and
        weights
        """
        from ase.io.conquest import read_bands, get_fermi_level, get_k_points

        with open(cq_out_path, 'r') as cqout_fileobj:
            self.eFermi = get_fermi_level(cqout_fileobj)

        with open(cq_out_path, 'r') as cqout_fileobj:
            self.eigenvalues, self.occupancies = \
                read_bands(self.parameters['nspin'], cqout_fileobj)

        with open(cq_out_path, 'r') as cqout_fileobj:
            self.kpoints, self.weights = get_k_points(cqout_fileobj)

    # Below functions necessary for ASE band_structure()
    def get_ibz_k_points(self):
        return self.kpoints

    def get_k_point_weights(self):
        return self.weights

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.eigenvalues[spin][kpt][:] * Hartree

    def get_fermi_level(self):
        return self.eFermi

    def get_number_of_spins(self):
        for key in self.parameters:
            if (key == 'nspin'):
                self.nspin = self.parameters[key]

        if (self.nspin is None):
            conquest_warn('nspin not specified in Conquest input')
            self.nspin = 1

        return self.nspin
