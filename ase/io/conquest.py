"""
This module contains functions to write all CONQUEST input files and
read all CONQUEST output files.
"""
import re
import os
import subprocess

from inspect import currentframe, getframeinfo

import numpy as np
from pathlib import Path
from shutil import move, copy

import ase.utils

from ase.units import Bohr, Hartree
from ase.constraints import FixAtoms, FixScaled
from ase.data import atomic_masses, atomic_numbers
from ase.calculators.calculator import ReadError
from ase.dft.kpoints import special_paths, sc_special_points, parse_path_string
from ase.symbols import Symbols
from ase.geometry import is_orthorhombic


class error(Exception):
    """Base class for exceptions in this module
    """
    pass


class warning(Warning):
    """Base class for warning in this module
    """
    pass


class conquest_err(error):
    """Exceptions related to Conquest I/O

    Attributes
    ----------
    message : explanation of the error
    """
    def __init__(self, message):
        self.message = message


class conquest_warn(warning):

    def __init__(self, message):
        self.message = message

        frameinfo = getframeinfo(currentframe().f_back)

        print('## ConquestWarning ##')
        print('file:', frameinfo.filename, ', line:', frameinfo.lineno)
        print('>>>> ', message)


class ConquestEnv:
    """Environmental variables for Conquest

    Attributes
    ----------
    get : return the named environmental variable
    warn : print a warning if this variable is not set
    """

    def __init__(self, stop=True):
        self._variables = {
            'cq_command': 'ASE_CONQUEST_COMMAND',
            'pseudo_path': 'CQ_PP_PATH',
            'gen_basis_command': 'CQ_GEN_BASIS_CMD'
        }
        for var in self._variables:
            if stop:
                self._error(self._variables[var])
            else:
                self._warn(self.variables[var])

    def get(self, var):
        return os.environ[self._variables[var]]

    def _warn(self, var):
        if var not in os.environ:
            print("Warning: env {} not found".format(var))

    def _error(self, var):
        if var not in os.environ:
            raise conquest_err("Env {} not set".format(var))

    def run_command(self, command):
        errorcode = subprocess.call(command, shell=True, stdout=subprocess.PIPE)
        if errorcode:
            raise RuntimeError(
                '{} returned an error: {}'.format(command, errorcode))


class EnvError(Exception):
    """Exceptions related to test environment

    Attributes
    ----------
    message : explanation of the error
    """

    def __init__(self, message):
        self.message = message


def read_conquest_out(fileobj, atoms):
    """
    Parse energy, forces and stresses from the Conquest_out file
    Returns results :: dict
    """
# WARNING: format change into release XXX
# ***old format pre-release
#    nspec_re = re.compile(
#        r'The number of atomic species in the system is.*?(\d+)(.*?)' +
#        r'Energy tolerance required', re.S | re.M)
#
# ***new format:
    nspec_re = re.compile(
        r'Number of species.*?(\d+)(.*?)' +
        r'end of species report', re.S | re.M)
# END WARNING

    # (?s:.*) forces search from the *end* of the string
    energy_re = re.compile(r'(?s:.*)DFT total energy\s+=\s+([-]?\d+\.\d+) Ha')

    force_re = re.compile(
        r'(?s:.*)Atom\s+X\s+Y\s+Z\s+(.*?)\s+end of force report', re.S | re.M)

    stress_re = re.compile(r'(?s:.*)Total stress:\s+(.*?)\s+GPa')

    num_atoms = len(atoms)
    forces = np.empty([num_atoms, 3])
    stresses = np.zeros(6)
    nspec = 0
#
#

#
#
#    test_re = re.compile(r'The number of atomic species in the system
#    is.*?(\d+)', re.S | re.M)
#    test_text = fileobj.read()
#    res = re.search(test_re, test_text)
#    print('res', res)
#    print(res.group(0))
#    print(res.group(1),len(res.group(1)))
#
#    fileobj.seek(0)
##
    text = fileobj.read()
    m = re.search(nspec_re, text)

    if not m:
        raise conquest_err("Could not find number of species in Conquest_out")
    nspec = int(m.group(1))
    specinfo = [line.strip() for line in m.group(2).strip().splitlines()]
    specinfo = [line for line in specinfo if not line == "PAO basis"]
    specinfo = specinfo[3:]

    order = []
    for n in range(nspec):
        # WARNING: format change into release XXXX
        # ***old format pre-release
        #        index, spec, mass, charge, rcut, nsf = specinfo[n].split()
        # ***new format:
        # print(specinfo[n].split())

        tmp1, index, mass, charge, rcut, nsf, spec, tmp2 = specinfo[n].split()
        # END WARNING

        order.append(spec)

    m = re.search(energy_re, text)
    if not m:
        raise conquest_err("Could not find DFT total energy in Conquest_out")
    energy = float(m.group(1))

    m = re.search(force_re, text)
    if not m:
        raise conquest_err("Could not find forces in Conquest_out")
    f = m.group(1).splitlines()

    for atom in range(num_atoms):
        for i, force in enumerate(f[atom].split()[2:]):
            forces[atom, i] = float(force)

    m = re.search(stress_re, text)
    if not m:
        raise conquest_err("Could not find stresses in Conquest_out")
    stresses[0:3] = np.array([float(bit) for bit in m.group(1).split()])
    energy = energy * Hartree
    force = forces * Hartree / Bohr
    stresses = stresses / 160.2176621
    stresses = stresses * Hartree / (atoms.get_cell().trace())

    return {'energy': energy, 'forces': force, 'stress': stresses}


def write_conquest(fileobj, atoms, atomic_order, fractional=True):
    """
    Write structure to CONQUEST-formatted file.
    """
    # LAT: not needed / use ase.geometry.orthorhombic instead
    #
    # orthorhombic = True
    # small = 1.0E-3
    # angles = atoms.get_cell_lengths_and_angles()[3:6]
    # for i in range(3):
    #     if ( abs(angles[i] - 90.0) >= small ):
    #         orthorhombic = False
    # assert orthorhombic, "Conquest can only handle orthorhombic cells"
    #
    # test_cell = atoms.get_cell()
    #
    # Test if orthorhombic / stop if not
    if (not is_orthorhombic(atoms.get_cell())):
        raise conquest_err("Conquest can only handle orthorhombic cells")

    #cellpar = atoms.get_cell_lengths_and_angles()
    cellpar = atoms.cell.cellpar()
    
    # CONQUEST by default works in units of Bohr:
    cellpar[0:3] = cellpar[0:3] / Bohr
    latfmt = '{0:<16.6f}{1:<16.6f}{2:<16.6f}\n'
    fileobj.write(latfmt.format(cellpar[0], 0., 0.))
    fileobj.write(latfmt.format(0., cellpar[1], 0.))
    fileobj.write(latfmt.format(0., 0., cellpar[2]))
    # RasMol complains if the atom index exceeds 100000. There might
    # be a limit of 5 digit numbers in this field.
    if atoms.constraints:
        moveflags = np.zeros((len(atoms), 3), dtype=bool)
        for constr in atoms.constraints:
            if isinstance(constr, FixScaled):
                moveflags[constr.a] = constr.mask
            elif isinstance(constr, FixAtoms):
                moveflags[constr.index] = [True, True, True]
    symbols = atoms.get_chemical_symbols()
    natoms = len(symbols)
    fileobj.write(f'{natoms}\n')
    coordfmt = '{0:>16.12f}{1:>16.12f}{2:>16.12f}{3:>4d}{4:>2s}{5:>2s}{6:>2s}\n'
    if fractional:
        p = atoms.get_scaled_positions()
    else:
        p = atoms.get_positions() / Bohr
    for a in range(natoms):
        if atoms.constraints:
            fixed = []
            for flag in moveflags[a]:
                if flag:  # True that it's fixed
                    fixed.append('F')
                else:
                    fixed.append('T')
        else:
            fixed = ['T', 'T', 'T']
        x, y, z = p[a]
        fileobj.write(coordfmt.format(
            x, y, z, 1 + atomic_order.index(symbols[a]),
            fixed[0], fixed[1], fixed[2]))


def setup_basis(species, basis=None, version="v323", xc="PBE",
                pot_file=None, vkb_file=None):
    """
    Setup basis set parameters to be added to the Conquest_input
    These appear in the species blocks.

    string  ::  species      atomic species. e.g. "He"
    string  ::  basis_size   use a predefined basis. can be one of:
                                "minimal", "small", "medium", "large"
    dict    ::  basis        if using anything other than the default basis_size
                                then this must be set.
                                Example: TODO
    string  ::  pot_file     ONCVPSP input file for the pseudopotential (.in)
    string  ::  vkb_file     Pseudopotential VKB file (.pot)
    """
    special = ['gen_basis', 'basis_size', 'pseudopotential_type']
    cq_env = ConquestEnv()
    path = Path(cq_env.get('pseudo_path')).joinpath(Path(xc + '/' + species))

    if not pot_file or not vkb_file:
        pot_file = path.joinpath(species + '.pot')
        in_file = path.joinpath(species + '.in')
    # Does the pseudo exist for this species?
    if not pot_file.is_file():
        raise ReadError('CQ pseudo file ' + str(pot_file) + ' missing.')
    if not in_file.is_file():
        raise ReadError('CQ pseudo file ' + str(in_file) + ' missing.')

    basis_str = ""
    basis_str += cqip_line("atom.pseudopotentialfile", str(in_file))
    basis_str += cqip_line("atom.vkbfile", str(pot_file))
    basis_str += cqip_line('atom.basissize', basis["basis_size"])

    for key in basis:
        if ((key not in special) and (key != 'xc')):
            basis_str += cqip_line(key, basis[key])

    return basis_str


def cqip_line(cq_key, value):
    '''
    Ensure different datatypes are correctly formated for Conquest input
    '''

    cq_input = '{0:<30}   '.format(cq_key)
    if isinstance(value, bool):  # isinstance evaluates bool as integer too!
        if value:
            cq_input += '{0}\n'.format("True")
        else:
            cq_input += '{0}\n'.format("False")
    elif isinstance(value, int) or isinstance(value, np.int64):
        cq_input += '{0}\n'.format(value)
    elif isinstance(value, float) or isinstance(value, np.float64):
        if value > 1000. or value < 1:
            cq_input += '{0:6.6e}\n'.format(value)  # Scientific notation
        else:
            cq_input += '{0:6.6f}\n'.format(value)  # round to 6 decimal places
    elif isinstance(value, str):
        cq_input += '{0}\n'.format(value)
    else:
        print("Warning - this could be wrong! Key: ", cq_key, " Value: ", value)
        cq_input += '{0}\n'.format(value)

    return cq_input


def make_ion_files(basis, species_list, command=None, directory=None, xc=None):
    """
    This is a preprocessing step that generates the basis sets (.ion files).

    dict    ::  basis        parameters for basis sets
    species_list :: list     *ordered* list of atomic species
    string  ::  command      path to gen_basis executable; otherwise set by
    environment variable CQ_GEN_BASIS_CMD

    """
    # TODO: need to rewrite since no more species_list ; single species

    cq_env = ConquestEnv()
    nspec = len(species_list)
    makeion_input = Path("Conquest_ion_input")
    makeion_input_spec = Path("Conquest_ion_input" + "_" + species_list[0])

    basis_strings = {}
    labels_string = '%block SpeciesLabels\n'
    i = 1
    for species in species_list:
        # TODO include the ability to specify an arbitrary ion file name!
        # We could do this by generalising 'species'
        if 'gen_basis' in basis[species]:
            if 'basis_size' not in basis[species]:
                print('basis_size not specified in basis for %s' % (species))
                print('Generating default basis (medium)')
                basis[species]['basis_size'] = 'medium'
            # get basic string - required to generate basis
            # cq_env.get('pseudo_path')

            basis_strings[species] = setup_basis(species, basis=basis[species],
                                                 xc=xc)
            # Write basic Conquest_input file in order to generate basis:
            labels_string += f'{i} {species}\n'
            i += 1

    labels_string += '%endblock\n'

    with ase.utils.workdir(directory):
        with makeion_input.open(mode='w') as fileobj:
            # Input file for basis generation tool MakeIonFiles
            fileobj.write(80 * '#' + '\n')
            fileobj.write('# CONQUEST input file\n')
            fileobj.write('# Only for basis generation!\n')
            fileobj.write(80 * '#' + '\n\n')
            fileobj.write(cqip_line('general.numberofspecies', nspec))
            fileobj.write(cqip_line('io.plotoutput', False))
            fileobj.write('\n')
            fileobj.write(labels_string)
            for species in species_list:
                fileobj.write(f'%block {species}\n')
                fileobj.write(basis_strings[species])
                fileobj.write('%endblock\n\n')

        copy(makeion_input, makeion_input_spec)

        # generate the basis sets
        # TODO: method of overriding ion_params?
        # TODO: check that pseudo types are all the same
        if not command:
            command = cq_env.get('gen_basis_command')

        cq_env.run_command(command)

        for species in species_list:
            ion_name_orig = species + 'CQ.ion'
            ion_name_new = species + '.ion'
            move(ion_name_orig, ion_name_new)


def get_basis_strings(species_list, basis, directory='.'):

    basis_strings = {}
    skip = ['basis_size', 'gen_basis', 'pseudopotential_type', 'file']
    if directory:
        olddir = os.getcwd()
        os.chdir(directory)

    for species in species_list:
        path = species + ".ion"

        ion_params = parse_ion(species, path)
        basis_strings[species] = ""
        for key in basis[species]:
            if key not in skip:
                basis_strings[species] += cqip_line(key, basis[species][key])

        for key in ion_params:
            if key not in basis[species]:
                basis_strings[species] += cqip_line(key, ion_params[key])

    if directory:
        os.chdir(olddir)

    return basis_strings


def parse_ion(species, ionfile):
    """
    Author: Jack Poulton, edited by J. K. Shenton
    This method parses the .ion file for the input species,
    and returns a dictionary of the following variables:
        valence_charge
        npao: assume that NSF=NPAO, for now...
        max_rcut: set the support_fn_range to a tiny bit more than this
        pseudopotential_type

    """
    ion_params = dict()
    # Check basis set type: siesta or hamann
    # TODO this could be made more clever

    with open(ionfile, 'r') as cqion:
        for line in cqion:
            # pseudopotential_type:
            if re.search("Hamann", line):
                pseudopotential_type = "hamann"
            if re.search("Troullier-Martins", line):
                pseudopotential_type = "siesta"

    npao = 0
    paoranges = []
    # Loop through the ion file:
    if pseudopotential_type == "hamann":
        with open(ionfile, 'r') as cqion:
            for line in cqion:
                # valence charge
                if re.search("Valence", line):
                    valence = float(line.split()[0])
                # cut off radii
                if re.search("Radii:", line):
                    radii = (line.split())
                    paorange = radii[1]
                    # return last line as largest orbital
                    paoranges.append(float(paorange))

                if re.search("population", line):
                    orbitals = line.split()
                    l = float(orbitals[0])
                    npao += 2 * l + 1

    elif pseudopotential_type == "siesta":
        re_cutoffs = re.compile(r'rcs')
        re_orbitals = re.compile(r'#orbital')

        paos = []
        paoranges = []

        with open(ionfile, 'r') as fileobj:
            for line in fileobj:
                if re_cutoffs.search(line):
                    paoranges.append(float(line.split()[1]))
                if re_orbitals.search(line):
                    l, n, z, is_polarised, population = line.split()[0:5]
                    paos.append([int(l), int(n), int(z),
                                int(is_polarised), float(population)])
        paos = np.array(paos)
        # npaos = sum(2l+1)
        npao = 0
        for pao in paos:
            npao += (2 * pao[0] + 1)
        # valence is sum of population:
        valence = sum(paos.T[4])

    eps_rc = 0.1  # The amount to add to the max PAO cut-off radius:
    ion_params["atom.valencecharge"] = valence
    ion_params["atom.numberofsupports"] = int(npao)
    ion_params["atom.supportfunctionrange"] = max(paoranges) + eps_rc

    return {k.lower(): ion_params[k] for k in ion_params}


def write_conquest_input(fileobj, atoms, atomic_order, parameters,
                         directory='.', basis={}):
    """
    Write input parameters to Conquest_input file.

    Parameters
    ==========

    fileobj :: file object
        Path to the Conquest_input file to be written

    atoms :: atom object
        The ASE atom object for the calculation

    atomic_order :: list of strings
        The CONQUEST structure file contains integer indices instead of atomic
        species, so the species must be supplied in the correct order in
        the read and write functions.
        e.g. atomic_order=['Bi','Fe','O']

    parameters :: dict
        Contains mandatory flags plus other CONQUEST flags as key/value pairs
    parameters = {'grid_cutoff'     : 100,
                  'xc'              : 'PBE',
                  'self_consistent' : True,
                  'scf_tolerance'   : 1.0e-6,
                  'kpts'            : None,
                  'nspin'           : 1,
                  'directory'       : None}

    basis :: dict
        A dictionary specifying the basis set parameters. These will
        generally be parsed from the .ion file, but can be specificed as in
        ase/calculators/conquest.py
    """

    # Translation of ASE keys into Conquest XC functionals
    cq_xc_dict = {'PZ': 1,    # Perdew-Zunger 81 LDA
                  'LDA': 3,    # Perdew-Wang 92 LDA
                  'PBE': 101,  # Perdew, Burke, Ernzerhof
                  'WC': 104   # Wu-Cohen
                  }
    cq_input = []
    for key in parameters:
        # special cases
        if key == 'grid_cutoff':
            cq_input.append(cqip_line("grid.gridcutoff",
                                      parameters['grid_cutoff']))
        elif key == 'xc':
            cq_input.append(cqip_line("general.functionaltype",
                                      cq_xc_dict[parameters['xc']]))

        elif key == 'self_consistent':
            cq_input.append(cqip_line("mine.selfconsistent",
                                      parameters['self_consistent']))

        elif key == 'scf_tolerance':
            cq_input.append(cqip_line("mine.sctolerance",
                                      parameters['scf_tolerance']))
        elif key == 'kpts':
            kpt_string = write_kpoints(atoms, parameters['kpts'])

        elif key == 'nspin':
            polarized = (parameters['nspin'] == 2)
            cq_input.append(cqip_line('spin.spinpolarised', polarized))

        elif key == 'directory':
            pass

        # all other keywords
        else:
            cq_input.append(cqip_line(key, parameters[key]))
    # - SPIN - # TODO initialise spins? This requires a lot more work!
    #        magmoms = atoms.get_initial_magnetic_moments()

    # - Species properties- #
    # TODO set these to the ones read from the .ion file for consistency
    masses = [atomic_masses[atomic_numbers[spec]] for spec in atomic_order]

    # - Chemical species block -
    chem_spec_block = '\n%block ChemicalSpeciesLabel\n'
    for i, spec in enumerate(atomic_order):
        chem_spec_block += f'{i+1: 2d} {masses[i]:8.3f} {spec}\n'
    chem_spec_block += '%endblock ChemicalSpeciesLabel\n'

    # - Species Blocks -
    basis_strings = get_basis_strings(atomic_order, basis, directory=directory)

    # - Number of species - #
    # TODO wrong if antiferromagnetism !
    num_species = len(atomic_order)
    if "general.numberofspecies" not in parameters:
        cq_input.append(cqip_line("general.numberofspecies", num_species))

    # Write to Conquest_input
    fileobj.write(80 * '#')
    fileobj.write('\n# CONQUEST Input file\n')
    fileobj.write('# Created using the Atomic Simulation Environment (ASE)\n')
    fileobj.write(80 * '#')
    fileobj.write('\n\n')
    cq_input.sort()
    for o in cq_input:
        fileobj.write(o)
    fileobj.write(kpt_string)
    fileobj.write(chem_spec_block)
    for spec in atomic_order:
        fileobj.write(f'\n%block {spec}\n')
        fileobj.write(basis_strings[spec])
        fileobj.write('%endblock\n')


def write_kpoints(atoms, kpts):
    """
    Write the part of Conquest_input where the k-points are specified.
    Generates either a Monkhorst-Pack grid or a set of points with weights.
    """
    kpt_string = "\n"
    if isinstance(kpts, list) or isinstance(kpts, np.ndarray):
        kpts = np.array(kpts)
        kpt_shape = kpts.shape
        if kpt_shape == (3,):  # assume MP mesh (x y z):
            kpt_string += cqip_line("diag.mpmesh", True)
            kpt_string += cqip_line("diag.mpmeshx", kpts[0])
            kpt_string += cqip_line("diag.mpmeshy", kpts[1])
            kpt_string += cqip_line("diag.mpmeshz", kpts[2])
        elif kpt_shape[-1] == 4 or (kpt_shape[-1] == 3):
            # Assume specifying coords (and weights)
            kpt_string += cqip_line("Diag.NumKpts", len(kpts))
            kpt_string += '%block Diag.Kpoints\n'
            for k in kpts:
                if kpt_shape[-1] == 3:  # fractional x, y, z, equal weighting
                    w = 1. / len(kpts)
                    kpt_string += \
                        f'{k[0]:2.6f} {k[1]:2.6f} {k[2]:2.6f} {w:2.4f}\n'
                else:  # fractional x, y, z, and the weight
                    kpt_string += \
                        f'{k[0]:2.6f} {k[1]:2.6f} {k[2]:2.6f {k[3]:2.4f}\n}'
            kpt_string += '%endblock Diag.Kpoints\n\n'
    elif isinstance(kpts, dict):
        # write a k-point path
        kpt_string += cqip_line("Diag.KspaceLines", True)
        # Symmetry of the cell
        tolerance = 4  # number of decimal places they have to agree to
        a, b, c = atoms.cell.lengths()
        a = round(a, tolerance)
        b = round(b, tolerance)
        c = round(c, tolerance)
        if a == b == c:
            symm = 'cubic'
        elif a == b != c or a != b == c or a == c != b:
            symm = 'tetragonal'
        elif a != b and b != c and a != c:
            symm = 'orthorhombic'

        # Path through k-space:
        if 'path' in kpts:
            if not isinstance(kpts['path'], str):
                raise ReadError("For k-point line mode, please pass the " +
                                "path as a string. e.g. {'path' : 'GXMGY'}")
            else:
                kpoint_path_string = kpts['path']
        else:
            # assume default k-point path
            kpoint_path_string = special_paths[symm]

        # what do the path labels correspond to in fractional recip. coords?
        if 'points' in kpts:
            # should be of the form: points : {"G" : [0, 0, 0],
            #                                  "X" : [1, 0, 0]}
            special_kpoints = kpts['points']
        else:
            # assume default special points for this symmetry
            special_kpoints = sc_special_points[symm]

        # Number of kpoints between each pair of high symmetry points
        if 'npoints' in kpts:
            npoints = kpts['npoints']
        else:
            # default to 25 k-points between each point of high
            npoints = 25

        num_pairs_points = sum([len(p) - 1 for p in
                                parse_path_string(kpoint_path_string)])
        kpt_string += cqip_line("diag.numkptlines", num_pairs_points)
        kpt_string += cqip_line("diag.numkpts", npoints)
        # Write the sequence of coords
        kpt_string += '\n%block Diag.KpointLines\n'
        for point_set in parse_path_string(kpoint_path_string):
            kpt_string += '#  {0}\n'.format(point_set)
            # for point in point_set:
            for ik in range(len(point_set) - 1):
                k_r = special_kpoints[point_set[ik]]
                kpt_string += f'{k_r[0]:2.6f} {k_r[1]:2.6f} {k_r[2]:2.6f}\n'
                k_r_2 = special_kpoints[point_set[ik + 1]]
                kpt_string += f'{k_r_2[0]:2.6f} {k_r_2[1]:2.6f} \
                                {k_r_2[2]:2.6f}\n'
        kpt_string += '%endblock\n'

    return kpt_string


def get_fermi_level(fileobj):
    """
    Parser the Fermi level from Conquest_out
    """

    text = fileobj.read()

    fermi_re = re.compile(r'Fermi energy for spin = (\d) ' +
                          r'is\s+([-]?\d+\.\d+) Ha')
    m = re.findall(fermi_re, text)
    if m:
        efermi = float(m[-1][1]) * Hartree

    return efermi


def get_k_points(fileobj):
    """
    Parse the ibz kpoints and weights from the Conquest_out file.
    Returns tuple of np arrays (kpoints, weights)
    """

    text = fileobj.read()

    ibz_kpts_re = re.compile(
        r'(\d+) symmetry inequivalent Kpoints in fractional ' +
        'coordinates:(.*?)\n\n', re.S | re.M)

    all_kpts_re = re.compile(
        r'All\s+(\d+)\s+Kpoints in fractional coordinates:' +
        '(.*?)\n\n', re.S | re.M)

    kpoints = []
    weights = []
    m = re.search(ibz_kpts_re, text)
    if not m:
        m = re.search(all_kpts_re, text)

    nkpts = int(m.group(1))
    kpts = m.group(2).strip().split('\n')
    for line in range(nkpts):
        i, kx, ky, kz, weight = kpts[line].strip().split()
        kpoints.append([float(kx), float(ky), float(kz)])
        weights.append(float(weight))

    return np.array(kpoints), np.array(weights)


def read_bands(nspin, fileobj):
    """
    Parse the Conquest_out file, grabbing the eigenvalues and k-points
    from a band structure calculation.

    Authors Jack Poulton & J Kane Shenton
    """

    nkpt_frac_re = \
        re.compile(r'(\d+)\s+symmetry inequivalent Kpoints in fractional')

    all_kpts_re = re.compile(
        r'All\s+(\d+)\s+Kpoints in fractional coordinates:' +
        '(.*?)\n\n', re.S | re.M)

    nkpt_cart_re = \
        re.compile(r'(\d+)\s+symmetry inequivalent Kpoints in Cartesian')

    kpt_block_1_re = \
        re.compile(r'Eigenvalues and occupancies(.*?)Sum of eigenvalues(.*?)\n',
                   re.M | re.S)
    kpt_block_2_re = \
        re.compile(
            r'Eigenvalues and occupancies(.*?)Sum of eigenvalues for ' +
            r'spin = 2.*?\n', re.M | re.S)
    eig_1_re = re.compile(
        r'for k-point\s+(\d+) :\s+([-]?\d+\.\d+\s+[-]?\d+\.\d+' +
        r'\s+[-]?\d+\.\d+)\n(.*)\n', re.M | re.S)
    eig_2_re = re.compile(
        r'for k-point\s+(\d+) :\s+([-]?\d+\.\d+\s+[-]?\d+\.\d+\s+' +
        r'[-]?\d+\.\d+)\s+For spin = 1\n(.*?)\s+Sum of eigenvalues.*?' +
        r'For spin = 2\n(.*)\n', re.M | re.S)

    text = fileobj.read()

    try:
        m = re.search(nkpt_frac_re, text)
        nkpoints = int(m.group(1))

    except AttributeError:
        try:
            m = re.search(nkpt_cart_re, text)
            nkpoints = int(m.group(1))

        except AttributeError:
            try:
                m = re.search(all_kpts_re, text)
                nkpoints = int(m.group(1))

            except AttributeError:
                print('re.search error!')

    if nspin == 1:
        kpt_blocks = re.findall(kpt_block_1_re, text)
    if nspin == 2:
        kpt_blocks = re.findall(kpt_block_2_re, text)
    # Eigenvalues printed out every scf cycle, only keep values for the last one
    kpt_blocks = kpt_blocks[-nkpoints:len(kpt_blocks)]
    eigenvalues = [[]]
    occupancies = [[]]
    if nspin == 2:
        eigenvalues.append([])
        occupancies.append([])
    for block in kpt_blocks:
        eigenvalues[0].append([])
        occupancies[0].append([])
        if nspin == 1:
            m = re.search(eig_1_re, block[0])
            eigs = re.split(r'\s\s\s+', m.group(3).strip(), flags=re.S | re.M)
            for i, pair in enumerate(eigs):
                eig, occ = [float(bit) for bit in pair.split()]
                eigenvalues[0][-1].append(eig)
                occupancies[0][-1].append(occ)
        if nspin == 2:
            eigenvalues[1].append([])
            occupancies[1].append([])
            m = re.search(eig_2_re, block)
            eigs_1 = re.split(r'\s\s\s+', m.group(3).strip(),
                              flags=re.S | re.M)
            eigs_2 = re.split(r'\s\s\s+', m.group(4).strip(),
                              flags=re.S | re.M)

            for i, pair in enumerate(eigs_1):
                eig_1, occ_1 = [float(bit) for bit in pair.split()]
                eig_2, occ_2 = [float(bit) for bit in eigs_2[i].split()]
                eigenvalues[0][-1].append(eig_1)
                occupancies[0][-1].append(occ_1)
                eigenvalues[1][-1].append(eig_2)
                occupancies[1][-1].append(occ_2)

    return np.array(eigenvalues), np.array(occupancies)


def sort_by_atomic_number(atoms):
    """
    Sort a list of atomic species by atomic number
    """
    sorted_numbers = np.array(sorted(set(atoms.numbers)))
    species_symbols = Symbols(sorted_numbers)
    return [species for species in species_symbols]
