"""Parsers for CASTEP .geom, .md, .ts files"""
from typing import Dict, Sequence, Optional, Union, TextIO

import numpy as np
from ase import Atoms
from ase.io.formats import string2index
from ase.utils import reader, writer


@reader
def read_castep_geom(fd, index=-1, units=None):
    """Reads a .geom file produced by the CASTEP GeometryOptimization task and
    returns an atoms  object.
    The information about total free energy and forces of each atom for every
    relaxation step will be stored for further analysis especially in a
    single-point calculator.
    Note that everything in the .geom file is in atomic units, which has
    been conversed to commonly used unit angstrom(length) and eV (energy).

    Note that the index argument has no effect as of now.

    Contribution by Wei-Bing Zhang. Thanks!

    Routine now accepts a filedescriptor in order to out-source the gz and
    bz2 handling to formats.py. Note that there is a fallback routine
    read_geom() that behaves like previous versions did.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    if isinstance(index, str):
        index = string2index(index)

    if units is None:
        from ase.io.castep import units_CODATA2002
        units = units_CODATA2002

    # fd is closed by embracing read() routine
    txt = fd.readlines()

    traj = []

    Hartree = units['Eh']
    Bohr = units['a0']

    for i, line in enumerate(txt):
        if line.find('<-- E') > 0:
            start_found = True
            energy = float(line.split()[0]) * Hartree
            cell = [x.split()[0:3] for x in txt[i + 1:i + 4]]
            cell = np.array([[float(col) * Bohr for col in row] for row in
                             cell])
        if line.find('<-- R') > 0 and start_found:
            start_found = False
            geom_start = i
            for i, line in enumerate(txt[geom_start:]):
                if line.find('<-- F') > 0:
                    geom_stop = i + geom_start
                    break
            species = [line.split()[0] for line in
                       txt[geom_start:geom_stop]]
            geom = np.array([[float(col) * Bohr for col in
                              line.split()[2:5]] for line in
                             txt[geom_start:geom_stop]])
            forces = np.array([[float(col) * Hartree / Bohr for col in
                                line.split()[2:5]] for line in
                               txt[geom_stop:geom_stop
                                   + (geom_stop - geom_start)]])
            image = Atoms(species, geom, cell=cell, pbc=True)
            # The energy in .geom or .md file is the force-consistent one
            # (possibly with the the finite-basis-set correction when, e.g.,
            # finite_basis_corr!=0 in GeometryOptimisation).
            # It should therefore be reasonable to assign it to `free_energy`.
            # Be also aware that the energy in .geom file not 0K extrapolated.
            image.calc = SinglePointCalculator(
                atoms=image, free_energy=energy, forces=forces)
            traj.append(image)

    return traj[index]


@reader
def read_castep_md(fd, index=-1, return_scalars=False, units=None):
    """Reads a .md file written by a CASTEP MolecularDynamics task
    and returns the trajectory stored therein as a list of atoms object.

    Note that the index argument has no effect as of now."""

    from ase.calculators.singlepoint import SinglePointCalculator

    if isinstance(index, str):
        index = string2index(index)

    if units is None:
        from ase.io.castep import units_CODATA2002
        units = units_CODATA2002

    factors = {
        't': units['t0'] * 1E15,     # fs
        'E': units['Eh'],            # eV
        'T': units['Eh'] / units['kB'],
        'P': units['Eh'] / units['a0']**3 * units['Pascal'],
        'h': units['a0'],
        'hv': units['a0'] / units['t0'],
        'S': units['Eh'] / units['a0']**3,
        'R': units['a0'],
        'V': np.sqrt(units['Eh'] / units['me']),
        'F': units['Eh'] / units['a0']}

    # fd is closed by embracing read() routine
    lines = fd.readlines()

    L = 0
    while 'END header' not in lines[L]:
        L += 1
    l_end_header = L
    lines = lines[l_end_header + 1:]
    times = []
    energies = []
    temperatures = []
    pressures = []
    traj = []

    # Initialization
    time = None
    Epot = None
    Ekin = None
    EH = None
    temperature = None
    pressure = None
    symbols = None
    positions = None
    cell = None
    velocities = None
    symbols = []
    positions = []
    velocities = []
    forces = []
    cell = np.eye(3)
    cell_velocities = []
    stress = []

    for (L, line) in enumerate(lines):
        fields = line.split()
        if len(fields) == 0:
            if L != 0:
                times.append(time)
                energies.append([Epot, EH, Ekin])
                temperatures.append(temperature)
                pressures.append(pressure)
                atoms = Atoms(
                    symbols=symbols,
                    positions=positions,
                    cell=cell,
                    pbc=True,
                )
                atoms.set_velocities(velocities)
                if len(stress) == 0:
                    atoms.calc = SinglePointCalculator(
                        atoms=atoms, free_energy=Epot, forces=forces)
                else:
                    atoms.calc = SinglePointCalculator(
                        atoms=atoms, free_energy=Epot,
                        forces=forces, stress=stress)
                traj.append(atoms)
            symbols = []
            positions = []
            velocities = []
            forces = []
            cell = []
            cell_velocities = []
            stress = []
            continue
        if len(fields) == 1:
            time = factors['t'] * float(fields[0])
            continue

        if fields[-1] == 'E':
            E = [float(x) for x in fields[0:3]]
            Epot, EH, Ekin = (factors['E'] * Ei for Ei in E)
            continue

        if fields[-1] == 'T':
            temperature = factors['T'] * float(fields[0])
            continue

        # only printed in case of variable cell calculation or calculate_stress
        # explicitly requested
        if fields[-1] == 'P':
            pressure = factors['P'] * float(fields[0])
            continue
        if fields[-1] == 'h':
            h = [float(x) for x in fields[0:3]]
            cell.append([factors['h'] * hi for hi in h])
            continue

        # only printed in case of variable cell calculation
        if fields[-1] == 'hv':
            hv = [float(x) for x in fields[0:3]]
            cell_velocities.append([factors['hv'] * hvi for hvi in hv])
            continue

        # only printed in case of variable cell calculation
        if fields[-1] == 'S':
            S = [float(x) for x in fields[0:3]]
            stress.append([factors['S'] * Si for Si in S])
            continue
        if fields[-1] == 'R':
            symbols.append(fields[0])
            R = [float(x) for x in fields[2:5]]
            positions.append([factors['R'] * Ri for Ri in R])
            continue
        if fields[-1] == 'V':
            V = [float(x) for x in fields[2:5]]
            velocities.append([factors['V'] * Vi for Vi in V])
            continue
        if fields[-1] == 'F':
            F = [float(x) for x in fields[2:5]]
            forces.append([factors['F'] * Fi for Fi in F])
            continue

    if return_scalars:
        data = [times, energies, temperatures, pressures]
        return data, traj[index]
    else:
        return traj[index]


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
