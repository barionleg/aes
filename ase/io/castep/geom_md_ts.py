"""Parsers for CASTEP .geom, .md, .ts files"""
from typing import Dict, Sequence, Optional, Union, TextIO

import numpy as np
from ase import Atoms
from ase.utils import writer


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
