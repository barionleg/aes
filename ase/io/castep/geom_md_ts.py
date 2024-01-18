"""Parsers for CASTEP .geom, .md, .ts files"""
from math import sqrt
from typing import Callable, Dict, List, Optional, Sequence, TextIO, Union

import numpy as np
from ase import Atoms
from ase.io.formats import string2index
from ase.utils import reader, writer


class Parser:
    """Parser for <-- `key` in .geom, .md, .ts files"""
    def __init__(self, units: Optional[Dict[str, float]] = None):
        if units is None:
            from ase.io.castep import units_CODATA2002
            self.units = units_CODATA2002
        else:
            self.units = units

    def parse(self, lines: List[str], key: str, method: Callable):
        """Parse <-- `key` in `lines` using `method`"""
        tmp = [_ for _ in lines if _.strip().endswith(key)]
        if len(tmp):
            return method(tmp, self.units)
        return None


@reader
def read_castep_geom(
    fd: TextIO,
    index: Union[int, slice, str] = -1,
    units: Optional[Dict[str, float]] = None,
):
    """Read a .geom file written by the CASTEP GeometryOptimization task

    Original contribution by Wei-Bing Zhang. Thanks!

    Parameters
    ----------
    fd : str | TextIO
        File name or object (possibly compressed with .gz and .bz2) to be read.
    index : int | slice | str, default: -1
        Index of image to be read.
    units : dict[str, float], default: None
        Dictionary with conversion factors from atomic units to ASE units.

        - ``Eh``: Hartree energy in eV
        - ``a0``: Bohr radius in Å
        - ``me``: electron mass in Da

        If None, values based on CODATA2002 are used.

    Returns
    -------
    Atoms | list[Atoms]
        ASE Atoms object or list of them.

    Notes
    -----
    The force-consistent energy, forces, stress are stored in ``atoms.calc``.

    Everything in the .geom file is in atomic units.
    They are converted in ASE units, i.e., Å (length), eV (energy), Da (mass).
    """
    if isinstance(index, str):
        index = string2index(index)
    if isinstance(index, str):
        raise ValueError(index)
    return list(iread_castep_geom(fd, units))[index]


def iread_castep_geom(fd: TextIO, units: Optional[Dict[str, float]] = None):
    """Read a .geom file of CASTEP GeometryOptimization as a generator"""
    parser = Parser(units)
    _read_header(fd)
    lines = []
    for line in fd:
        if line.strip():
            lines.append(line)
        else:
            yield _read_atoms_geom(lines, parser)
            lines = []


def _read_atoms_geom(lines: List[str], parser: Parser):
    from ase.calculators.singlepoint import SinglePointCalculator

    energy = parser.parse(lines, '<-- E', _read_energies)
    cell = parser.parse(lines, '<-- h', _read_cell)
    stress = parser.parse(lines, '<-- S', _read_stress)
    symbols, positions = parser.parse(lines, '<-- R', _read_positions)
    forces = parser.parse(lines, '<-- F', _read_forces)

    atoms = Atoms(symbols, positions, cell=cell, pbc=True)
    # The energy in .geom or .md file is the force-consistent one
    # (possibly with the the finite-basis-set correction when, e.g.,
    # finite_basis_corr!=0 in GeometryOptimisation).
    # It should therefore be reasonable to assign it to `free_energy`.
    # Be also aware that the energy in .geom file not 0K extrapolated.
    atoms.calc = SinglePointCalculator(
        atoms=atoms,
        free_energy=energy,
        forces=forces,
        stress=stress,
    )
    return atoms


@reader
def read_castep_md(
    fd: TextIO,
    index: Union[int, slice, str] = -1,
    units: Optional[Dict[str, float]] = None,
):
    """Read a .md file written by the CASTEP MolecularDynamics task

    Parameters
    ----------
    fd : str | TextIO
        File name or object (possibly compressed with .gz and .bz2) to be read.
    index : int | slice | str, default: -1
        Index of image to be read.
    units : dict[str, float], default: None
        Dictionary with conversion factors from atomic units to ASE units.

        - ``Eh``: Hartree energy in eV
        - ``a0``: Bohr radius in Å
        - ``me``: electron mass in Da

        If None, values based on CODATA2002 are used.

    Returns
    -------
    Atoms | list[Atoms]
        ASE Atoms object or list of them.

    Notes
    -----
    The force-consistent energy, forces, stress are stored in ``atoms.calc``.

    Everything in the .md file is in atomic units.
    They are converted in ASE units, i.e., Å (length), eV (energy), Da (mass).
    """
    if isinstance(index, str):
        index = string2index(index)
    if isinstance(index, str):
        raise ValueError(index)
    return list(iread_castep_md(fd, units))[index]


def iread_castep_md(fd: TextIO, units: Optional[Dict[str, float]] = None):
    """Read a .md file of CASTEP MolecularDynamics as a generator"""
    parser = Parser(units)
    _read_header(fd)
    lines = []
    for line in fd:
        if line.strip():
            lines.append(line)
        else:
            yield _read_atoms_md(lines, parser)
            lines = []


def _read_atoms_md(lines: List[str], parser: Parser):
    from ase.calculators.singlepoint import SinglePointCalculator

    factor = parser.units['a0'] * sqrt(parser.units['me'] / parser.units['Eh'])
    time = float(lines[0].split()[0]) * factor  # au -> ASE

    energy = parser.parse(lines, '<-- E', _read_energies)
    # temperature = parser.parse(lines, '<-- T', _read_temperature)
    # only printed in case of variable cell calculation or calculate_stress
    # pressure = parser.parse(lines, '<-- P', _read_pressure)
    cell = parser.parse(lines, '<-- h', _read_cell)
    # only printed in case of variable cell calculation
    # cell_velocities = parser.extract(lines, '<-- hv', _read_cell_velocities)
    # only printed in case of variable cell calculation
    stress = parser.parse(lines, '<-- S', _read_stress)
    symbols, positions = parser.parse(lines, '<-- R', _read_positions)
    velocities = parser.parse(lines, '<-- V', _read_velocities)
    forces = parser.parse(lines, '<-- F', _read_forces)

    atoms = Atoms(symbols, positions, cell=cell, pbc=True)
    atoms.info['time'] = time
    atoms.set_velocities(velocities)
    atoms.calc = SinglePointCalculator(
        atoms=atoms,
        free_energy=energy,
        forces=forces,
        stress=stress,
    )
    return atoms


def _read_header(fd: TextIO):
    for line in fd:
        if 'END header' in line:
            next(fd)  # read blank line below 'END header'
            break


def _read_energies(lines: List[str], units: Dict[str, float]):
    """Read force-consistent energy

    Notes
    -----
    Enthalpy and kinetic energy (in .md) are also written in the same line.
    They are however not parsed because they can be computed using stress and
    atomic velocties, respsectively.
    """
    return float(lines[0].split()[0]) * units['Eh']


def _read_temperature(lines: List[str], units: Dict[str, float]):
    """Read temperature

    Notes
    -----
    Temperature can be computed from kinetic energy and hence not necessary.
    """
    factor = units['Eh'] / units['kB']  # hartree -> K
    return float(lines[0].split()[0]) * factor


def _read_pressure(lines: List[str], units: Dict[str, float]):
    """Read pressure

    Notes
    -----
    Pressure can be computed from stress and hence not necessary.
    """
    factor = units['Eh'] / units['a0']**3  # au -> eV/A3
    return float(lines[0].split()[0]) * factor


def _read_cell(lines: List[str], units: Dict[str, float]):
    bohr = units['a0']
    cell = np.array([_.split()[0:3] for _ in lines], dtype=float)
    return cell * bohr


# def _read_cell_velocities(lines: List[str], units: Dict[str, float]):
#     hartree = units['Eh']
#     me = units['me']
#     cell_velocities = np.array([_.split()[0:3] for _ in lines], dtype=float)
#     return cell_velocities * np.sqrt(hartree / me)


def _read_stress(lines: List[str], units: Dict[str, float]):
    hartree = units['Eh']
    bohr = units['a0']
    stress = np.array([_.split()[0:3] for _ in lines], dtype=float)
    return stress.reshape(-1)[[0, 4, 8, 5, 2, 1]] * (hartree / bohr**3)


def _read_positions(lines: List[str], units: Dict[str, float]):
    bohr = units['a0']
    symbols = [_.split()[0] for _ in lines]
    positions = np.array([_.split()[2:5] for _ in lines], dtype=float)
    return symbols, positions * bohr


def _read_velocities(lines: List[str], units: Dict[str, float]):
    hartree = units['Eh']
    me = units['me']
    velocities = np.array([_.split()[2:5] for _ in lines], dtype=float)
    return velocities * np.sqrt(hartree / me)


def _read_forces(lines: List[str], units: Dict[str, float]):
    hartree = units['Eh']
    bohr = units['a0']
    forces = np.array([_.split()[2:5] for _ in lines], dtype=float)
    return forces * (hartree / bohr)


@writer
def write_castep_geom(
    fd: TextIO,
    images: Union[Atoms, Sequence[Atoms]],
    units: Optional[Dict[str, float]] = None,
    sort: bool = False,
):
    """Write a CASTEP .geom file.

    Parameters
    ----------
    fd : str | TextIO
        File name or object (possibly compressed with .gz and .bz2) to be read.
    images : Atoms | Sequenece[Atoms]
        ASE Atoms object(s) to be written.
    units : dict[str, float], default: None
        Dictionary with conversion factors from atomic units to ASE units.

        - ``Eh``: Hartree energy in eV
        - ``a0``: Bohr radius in Å
        - ``me``: electron mass in Da

        If None, values based on CODATA2002 are used.
    sort : bool, default: False
        If True, atoms are sorted in ascending order of atomic number.

    Notes
    -----
    Values in the .geom file are in atomic units.
    """
    if isinstance(images, Atoms):
        images = [images]

    if units is None:
        from ase.io.castep import units_CODATA2002
        units = units_CODATA2002

    _write_header(fd)

    for index, atoms in enumerate(images):
        if sort:
            atoms = atoms[atoms.numbers.argsort()]
        _write_convergence_status(fd, index)
        _write_energies_geom(fd, atoms, units)
        _write_cell(fd, atoms, units)
        _write_stress(fd, atoms, units)
        _write_positions(fd, atoms, units)
        _write_forces(fd, atoms, units)
        fd.write('  \n')


@writer
def write_castep_md(
    fd: TextIO,
    images: Union[Atoms, Sequence[Atoms]],
    units: Optional[Dict[str, float]] = None,
    sort: bool = False,
):
    """Write a CASTEP .md file.

    Parameters
    ----------
    fd : str | TextIO
        File name or object (possibly compressed with .gz and .bz2) to be read.
    images : Atoms | Sequenece[Atoms]
        ASE Atoms object(s) to be written.
    units : dict[str, float], default: None
        Dictionary with conversion factors from atomic units to ASE units.

        - ``Eh``: Hartree energy in eV
        - ``a0``: Bohr radius in Å
        - ``me``: electron mass in Da

        If None, values based on CODATA2002 are used.
    sort : bool, default: False
        If True, atoms are sorted in ascending order of atomic number.

    Notes
    -----
    Values in the .md file are in atomic units.
    """
    if isinstance(images, Atoms):
        images = [images]

    if units is None:
        from ase.io.castep import units_CODATA2002
        units = units_CODATA2002

    _write_header(fd)

    for index, atoms in enumerate(images):
        if sort:
            atoms = atoms[atoms.numbers.argsort()]
        _write_time(fd, index)
        _write_energies_md(fd, atoms, units)
        _write_temperature(fd, atoms, units)
        _write_cell(fd, atoms, units)
        _write_cell_velocities(fd, atoms, units)
        _write_stress(fd, atoms, units)
        _write_positions(fd, atoms, units)
        _write_velocities(fd, atoms, units)
        _write_forces(fd, atoms, units)
        fd.write('  \n')


def _format_float(x: float) -> str:
    """Format a floating number for .geom and .md files"""
    return np.format_float_scientific(
        x,
        precision=16,
        unique=False,
        pad_left=2,
        exp_digits=3,
    ).replace('e', 'E')


def _write_header(fd: TextIO):
    fd.write(' BEGIN header\n')
    fd.write('  \n')
    fd.write(' END header\n')
    fd.write('  \n')


def _write_convergence_status(fd: TextIO, index: int):
    fd.write(21 * ' ')
    fd.write(f'{index:18d}')
    fd.write(34 * ' ')
    for _ in range(4):
        fd.write('   F')  # Convergence status. So far F for all.
    fd.write(10 * ' ')
    fd.write('  <-- c\n')


def _write_time(fd: TextIO, index: int):
    fd.write(18 * ' ' + f'   {_format_float(index)}\n')  # So far index.


def _write_energies_geom(fd: TextIO, atoms: Atoms, units: Dict[str, float]):
    hartree = units['Eh']
    if atoms.calc is None:
        return
    if atoms.calc.results.get('free_energy') is None:
        return
    energy = atoms.calc.results.get('free_energy') / hartree
    fd.write(18 * ' ')
    fd.write(f'   {_format_float(energy)}')
    fd.write(f'   {_format_float(energy)}')
    fd.write(27 * ' ')
    fd.write('  <-- E\n')


def _write_energies_md(fd: TextIO, atoms: Atoms, units: Dict[str, float]):
    hartree = units['Eh']
    if atoms.calc is None:
        return
    if atoms.calc.results.get('free_energy') is None:
        return
    energy = atoms.calc.results.get('free_energy') / hartree
    enthalpy = energy
    if atoms.calc is not None and atoms.calc.results.get('stress') is not None:
        pressure = np.mean(atoms.calc.results.get('stress')[[0, 1, 2]])
        volume = atoms.get_volume()
        enthalpy += pressure * volume / hartree
    kinetic = atoms.get_kinetic_energy() / hartree
    fd.write(18 * ' ')
    fd.write(f'   {_format_float(energy)}')
    fd.write(f'   {_format_float(enthalpy)}')
    fd.write(f'   {_format_float(kinetic)}')
    fd.write('  <-- E\n')


def _write_temperature(fd: TextIO, atoms: Atoms, units: Dict[str, float]):
    hartree = units['Eh']
    kinetic = atoms.get_kinetic_energy() / hartree
    fd.write(18 * ' ')
    fd.write(f'   {_format_float(kinetic)}')
    fd.write(54 * ' ')
    fd.write('  <-- T\n')


def _write_cell(fd: TextIO, atoms: Atoms, units: Dict[str, float]):
    bohr = units['a0']
    cell = atoms.cell / bohr  # in bohr
    for i in range(3):
        fd.write(18 * ' ')
        for j in range(3):
            fd.write(f'   {_format_float(cell[i, j])}')
        fd.write('  <-- h\n')


def _write_cell_velocities(fd: TextIO, atoms: Atoms, units: Dict[str, float]):
    pass  # TODO: to be implemented


def _write_stress(fd: TextIO, atoms: Atoms, units: Dict[str, float]):
    hartree = units['Eh']
    bohr = units['a0']
    if atoms.calc is None:
        return
    if atoms.calc.results.get('stress') is None:
        return
    stress = atoms.calc.results.get('stress') / (hartree / bohr**3)  # Voigt
    stress = stress[[0, 5, 4, 5, 1, 3, 4, 3, 2]].reshape(3, 3)  # matrix
    for i in range(3):
        fd.write(18 * ' ')
        for j in range(3):
            fd.write(f'   {_format_float(stress[i, j])}')
        fd.write('  <-- S\n')


def _write_positions(fd: TextIO, atoms: Atoms, units: Dict[str, float]):
    bohr = units['a0']
    positions = atoms.positions / bohr
    symbols = atoms.symbols
    indices = symbols.species_indices()
    for i, symbol, position in zip(indices, symbols, positions):
        fd.write(f' {symbol:8s}')
        fd.write(f' {i + 1:8d}')
        for j in range(3):
            fd.write(f'   {_format_float(position[j])}')
        fd.write('  <-- R\n')


def _write_forces(fd: TextIO, atoms: Atoms, units: Dict[str, float]):
    hartree = units['Eh']
    bohr = units['a0']
    if atoms.calc is None:
        return
    if atoms.calc.results.get('forces') is None:
        return
    forces = atoms.calc.results.get('forces') / (hartree / bohr)
    symbols = atoms.symbols
    indices = symbols.species_indices()
    for i, symbol, force in zip(indices, symbols, forces):
        fd.write(f' {symbol:8s}')
        fd.write(f' {i + 1:8d}')
        for j in range(3):
            fd.write(f'   {_format_float(force[j])}')
        fd.write('  <-- F\n')


def _write_velocities(fd: TextIO, atoms: Atoms, units: Dict[str, float]):
    hartree = units['Eh']
    me = units['me']
    velocities = atoms.get_velocities() / np.sqrt(hartree / me)
    symbols = atoms.symbols
    indices = symbols.species_indices()
    for i, symbol, velocity in zip(indices, symbols, velocities):
        fd.write(f' {symbol:8s}')
        fd.write(f' {i + 1:8d}')
        for j in range(3):
            fd.write(f'   {_format_float(velocity[j])}')
        fd.write('  <-- V\n')
