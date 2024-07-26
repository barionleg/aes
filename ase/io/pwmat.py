import warnings
import io
from typing import List, Dict, Optional, Union
import numpy as np
import os
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointDFTCalculator
from ase.constraints import FixAtoms
from ase.data import chemical_symbols
from ase.io.pwmat_namelist.namelist import Namelist_pwmat
from ase.io.espresso import kspacing_to_grid
from ase.units import create_units
from ase.utils import reader, writer
from ase.stress import full_3x3_to_voigt_6_stress

units = create_units('2018')

_PWmat_Cell = 'Lattice vector (Angstrom)'
_PWmat_Stress = 'stress(eV)'
_PWmat_Position = 'Position'
_PWmat_Force = 'Force'
_PWmat_End = '----------'


@reader
def read_pwmat(fd: io.TextIOWrapper, ignore_constraints: bool = False) -> Atoms:
    """
    Read Atoms object from PWmat atom.config input/output file.

    Parameters
    ----------
    fd : io.TextIOWrapper

    ignore_constraints: bool
        Ignore all constraints on `atoms`, default to False.
    Returns
    -------
    atoms : Atoms
        Atoms object
    """
    natoms: int = int(next(fd).split()[0])
    while ("LATTICE" in next(fd).upper()):
        break
    lattice: List[List[float]] = []
    positions: List[List[float]] = []
    numbers: List[int] = []
    magmoms: List[float] = []
    fix_indices: List[int] = []
    for _ in range(3):
        line = next(fd)
        lattice.append([float(val) for val in line.split()])
    while ("POSITION" in next(fd).upper()):
        break
    for ind in range(natoms):
        line_lst = next(fd).split()
        numbers.append(int(line_lst[0]))
        tmp_frac_position = np.array([float(line_lst[1]), float(line_lst[2]),
                                      float(line_lst[3])])
        if int(line_lst[4]) != 1:
            fix_indices.append(ind)
        # tmp_cart_position =
        # list(np.dot(tmp_frac_position, lattice).reshape(3))
        positions.append(tmp_frac_position)
    for ii in fd:
        if ("MAGNETIC" in ii):
            for _ in range(natoms):
                magmoms.append(float(next(fd).split()[1]))
        break
    atoms = Atoms(cell=lattice, scaled_positions=positions,
                  numbers=numbers, magmoms=magmoms, pbc=(1, 1, 1))
    if not ignore_constraints:
        c = FixAtoms(indices=fix_indices)
        atoms.set_constraint(c)
    return atoms


@writer
def write_pwmat(fd: io.TextIOWrapper, atoms: Atoms, sort: bool = True,
                ignore_constraints: bool = False,
                show_magnetic: bool = False, **kwargs) -> None:
    """
    Writes atom into PWmat input format

    Args:
        fd (io.TextIOWrapper):
            File like object to which the atoms object should be written
        atoms (Atoms):
            Input structure
        sort (bool):
            Sort the atomic indices alphabetically by element. Defaults to True.
        ignore_constraints (bool):
            Ignore all constraints on `atoms`. Defaults to False.
        show_magnetic (bool):
            Whether to write MAGNETIC label in atom.config. Defaults to False.

    """
    if sort:
        index_order = sorted([id for id in range(len(atoms))],
                             key=lambda x: atoms.numbers[x])
        atoms = atoms[index_order]

    if not ignore_constraints:
        if atoms.constraints:
            for c in atoms.constraints:
                fix_indices = c.index
        else:
            fix_indices = []
    else:
        fix_indices = []

    fd.write(f'{len(atoms)}   atoms\n')
    fd.write('Lattice vector\n')
    for line in np.array(atoms.get_cell()):
        fd.write("%12.6f   %12.6f   %12.6f \n" % (line[0], line[1], line[2]))
    fd.write('Position, move_x, move_y, move_z\n')
    for i in range(len(atoms)):
        if i in fix_indices:
            fd.write('{}	{:18.15f}	{:18.15f}	{:18.15f}  0   0   0\n'.format
                     (atoms.numbers[i], *atoms.get_scaled_positions()[i]))
        else:
            fd.write('{}	{:18.15f}	{:18.15f}	{:18.15f}  1   1   1\n'.format
                     (atoms.numbers[i], *atoms.get_scaled_positions()[i]))
    if show_magnetic:
        fd.write('MAGNETIC\n')
        for j in range(len(atoms)):
            fd.write('{}	{:18.15f}\n'.format(atoms.numbers[j], atoms[j].magmom))


@reader
def read_pwmat_report(fd: io.TextIOWrapper, index=-1, MOVEMENT=None, QDIV=None):
    """
    Read potential energy and fermi energy from REPORT.
    Read positions, cells, stresses and forces from MOVEMENT (optional).
    Read magnetic moments from OUT.QDIV (optional).

    Args:
        fd: io.TextIOWrapper
            file object of REPORT file.
        index : int or slice or str
            Which frame(s) to read. The default is -1 (last frame).
            See :func:`ase.io.read` for details.
        MOVEMENT: Union[str, None], optional]
            Defaults to None. The path of MOVEMENT file.
        QDIV: Union[str, None], optional]
            Defaults to None. The path of OUT.QDIV file.

    """
    # energy and fermi energy
    energies = []
    efermis = []
    for line in fd:
        if 'E_Fermi(eV)' in line:
            efermis.append(float(line.strip().split()[-1]))
        if 'E_tot(eV)    =' in line:
            energies.append(float(line.split()[2]))

    if MOVEMENT is not None and os.path.exists(MOVEMENT):
        indexes: Dict[str, List[int]] = {
            _PWmat_Cell: [],
            _PWmat_Stress: [],
            _PWmat_Position: [],
            _PWmat_Force: [],
            _PWmat_End: []
        }
        lines = open(MOVEMENT).readlines()
        for i, line in enumerate(lines):
            for key in list(indexes):
                if key in line:
                    indexes[key].append(i)
        indexes[_PWmat_Force] = indexes[_PWmat_Force][1::2]
        assert len(energies) == len(indexes[_PWmat_Position]) - 1
        # cells and stresses
        cells = []
        stresses = []
        for ii in range(len(indexes[_PWmat_Cell])):
            line_start = indexes[_PWmat_Cell][ii] + 1
            line_end = indexes[_PWmat_Position][ii] - 1
            cell = []
            stress: Optional[Union[List[List[float]], None]] = []
            for line in lines[line_start:line_end + 1]:
                cell.append([float(l) for l in line.split()[:3]])
                if 'stress:' not in line:
                    stress = None
                else:
                    if stress is not None:
                        stress.append([float(l) for l in
                                       line.strip().split()[-3:]])
            cells.append(cell)
            if stress is not None:
                stress = full_3x3_to_voigt_6_stress(stress)
            stresses.append(stress)
        # symbols and positions
        symbols = []
        positions = []
        for jj in range(len(indexes[_PWmat_Position])):
            line_start = indexes[_PWmat_Position][jj] + 1
            line_end = indexes[_PWmat_Force][jj] - 1
            symbol = []
            position = []
            for line in lines[line_start:line_end + 1]:
                position.append([float(l) for l in line.split()[1:4]])
                symbol.append(chemical_symbols[int(line.split()[0])])
            positions.append(position)
            symbols.append(symbol)
        # forces
        forces = []
        for kk in range(len(indexes[_PWmat_Force])):
            line_start = indexes[_PWmat_Force][kk] + 1
            line_end = indexes[_PWmat_End][kk] - 1
            force = []
            for line in lines[line_start:line_end + 1]:
                force.append([float(l) for l in line.split()[1:4]])
            forces.append(force)
    else:
        # cells_none = None
        stresses_none = None
        # symbols_none = None
        # positions_none = None
        forces_none = None

    if QDIV is not None and os.path.exists(QDIV):
        lines = open(QDIV).readlines()
        lines.pop(0)
        magmoms = [float(line.strip().split()[-1]) for line in lines]
    else:
        magmoms = None

    images = []
    if MOVEMENT is not None and os.path.exists(MOVEMENT):
        for i_sp in range(len(energies)):
            atoms = Atoms(symbols=symbols[i_sp],
                          scaled_positions=positions[i_sp],
                          cell=cells[i_sp], pbc=True)
            calc = SinglePointDFTCalculator(atoms, energy=energies[i_sp],
                                            free_energy=energies[i_sp],
                                            forces=forces[i_sp],
                                            stress=stresses[i_sp],
                                            magmoms=magmoms,
                                            efermi=efermis[i_sp])
            atoms.calc = calc
            images.append(atoms)
    else:
        atoms = Atoms()
        try:
            calc = SinglePointDFTCalculator(atoms, energy=energies[-1],
                                            free_energy=energies[-1],
                                            forces=forces_none,
                                            stress=stresses_none,
                                            magmoms=magmoms, efermi=efermis[-1])
        except IndexError:
            warnings.warn('No energy value in REPORT file, \
and a fake value is assigned to energy.')
            calc = SinglePointDFTCalculator(atoms, energy=0.0, free_energy=0.0,
                                            forces=forces_none,
                                            stress=stresses_none,
                                            magmoms=magmoms, efermi=None)
        atoms.calc = calc
        images.append(atoms)
    if images:
        if isinstance(index, int):
            steps = [images[index]]
        else:
            steps = images[index]
    else:
        steps = []

    for step in steps:
        # step.set_initial_magnetic_moments(magmoms=magmoms)
        yield step


@writer
def write_pwmat_in(fd: io.TextIOWrapper, atoms: Atoms, input_data=None,
                   kspacing=None, **kwargs):
    """
    Generates etot.input file for PWmat calculation.

    Args:
        fd (io.TextIOWrapper):
            file object of etot.input file which the parameters
            should be written.
        atoms (Atoms):
            Structure for PWmat calculation.
        input_data (Dict):
            A dictionary with input parameters for PWmat calculation.
        kspacing (float):
            Generate a grid of k-points with this as the minimum distance
            in A^-1 between them in reciprocal space.

    Raises:
        KeyError: _description_
    """
    input_parameters = Namelist_pwmat(input_data)
    input_parameters.to_nested(**kwargs)
    if kspacing is not None:
        kgrid = kspacing_to_grid(atoms, kspacing)
        if 'MP_N123' in list(input_parameters):
            mp_n123_tmp = input_parameters['MP_N123'].split().copy()
            mp_n123_tmp[:3] = kgrid
            mp_n123_tmp = [str(t) for t in mp_n123_tmp]
            input_parameters['MP_N123'] = ' '.join(mp_n123_tmp)
        else:
            job_name = input_parameters.get('JOB', None)
            if job_name is not None:
                if job_name.upper() in ['SCF', 'NONSCF', 'DOS', 'MOMENT',
                                        'RELAX', 'EGGFIT', 'DIMER', 'SCFEP',
                                        'POTENTIAL', 'HPSI', 'WKM']:
                    mp_n123_tmp = kgrid + [0, 0, 0, 0]
                    mp_n123_tmp = [str(t) for t in mp_n123_tmp]
                    input_parameters['MP_N123'] = ' '.join(mp_n123_tmp)
                if job_name.upper() in ['MD', 'TDDFT', 'NAMD', 'NEB']:
                    mp_n123_tmp = kgrid + [0, 0, 0, 2]
                    mp_n123_tmp = [str(t) for t in mp_n123_tmp]
                    input_parameters['MP_N123'] = ' '.join(mp_n123_tmp)
            else:
                raise KeyError('The keyword JOB must be given !')
    else:
        if 'MP_N123' not in list(input_parameters):
            warnings.warn('The default value is used for the k-point.')
    spin = input_parameters.get('SPIN', None)
    mp_n123 = input_parameters.get('MP_N123', None)
    if spin is not None and mp_n123 is not None and int(spin) == 222:
        mp_n123_tmp = mp_n123.split().copy()
        mp_n123_tmp[-1] = '2'
        input_parameters['MP_N123'] = ' '.join(mp_n123_tmp)
    fd.write(input_parameters.to_string())


def write_IN_KPT(dir_path: str, atoms: Atoms, density=None, **kwargs):
    """_summary_

    Args:
        fd (str):
            directory of IN.KPT file
        atoms (Atoms):
            Structure for PWmat calculation.
        density (float):
            k-points per 1/A on the output kpts list.
             See :func: 'ase.dft.kpoints.bandpath' for details.
    """
    fd = open(dir_path, 'w+')
    lat = atoms.cell.get_bravais_lattice()
    # special_points = lat.get_special_points()
    band_path = atoms.cell.bandpath(lat.special_path, density=density)
    index2name = band_path._find_special_point_indices(eps=1e-5)
    kpts = band_path.kpts
    nkpts = len(kpts)

    fd.write('{:>5}\n'.format(nkpts))
    fd.write('{:>5}{:>5}\n'.format(2, 0))
    for i in range(nkpts):
        if i in list(index2name):
            fd.write('{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}{:>14}\n'.format
                     (*kpts[i], 1.0, index2name[i]))
        else:
            fd.write('{:>14.5f}{:>14.5f}{:>14.5f}{:>14.5f}\n'.format
                     (*kpts[i], 1.0))
    fd.close()
