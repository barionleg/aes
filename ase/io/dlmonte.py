import re
from numpy import zeros, isscalar, inf
from ase.data import chemical_symbols

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator

chemical_symbols.extend(['Me']) # quality of life helper for dummy atoms


def read_single_dlm_image(f, levcfg, imcon, natoms, is_trajectory, symbols=None, verbose=False):
    """
    :param f: IOWrapper object containing your ASCII DL_MONTE config file
    :param levcfg: Data types to be read in. Expects positional only if == 0
    :param imcon: Coordinate system. Cartesian if 0 or fractional if 1.
    :param natoms: number of expected atoms
    :param is_trajectory: Bool. Expects multiple frames if true.
    :param symbols: a list of atomtype symbols (?)
    :param verbose: Bool. Prints extra information if true
    :return: A dictionary of format {molecule name: Atoms object}
    """
    verboseprint = print if verbose else lambda *a, **k: None

    cell = zeros((3, 3))
    ispbc = imcon > 0
    if ispbc or is_trajectory:
        for j in range(3):
            line = f.readline()
            line = line.split()
            for i in range(3):
                try:
                    cell[j, i] = float(line[i])
                except ValueError:
                    raise RuntimeError("error reading cell")
    if symbols:
        sym = symbols
    else:
        sym = []

    positions = []
    velocities = []
    forces = []

    if is_trajectory:
        counter = natoms
    else:
        counter = inf  # clunky fix to handle single frame files (e.g. CONFIG)
    verboseprint('Atoms to read before stopping:', counter)
    labels = []

    molformat = False
    a = 0
    while line and (a < counter):
        # print(a)
        a += 1
        line = f.readline()
        if not line:
            a -= 1
            break

        m = re.match(r'\s*([A-Za-z][a-z]?)(\S*)', line)
        assert m is not None, line
        # print('line:',m.string.split()[0])
        if m.string.split()[0] == 'NUMMOL':  # catch for a NUMMOL statement
            nummol = line.split()[1]
            verboseprint('nummol detected:', line.split()[1])
            molformat = True  # outputs each molecule as a separate Atoms object
            molnames = {}  # dictionary of unique molecule names and lengths
            molnamelist = []  # list of molecule names in order
            molstart = []  # start positions for each molecule
            counter += 1
            continue
        if m.string.split()[0] == 'MOLECULE':  # catch new molecules
            verboseprint('Molecule detected:', line.split()[1])
            try:
                molnames
                molnamelist
                molstart
            except NameError:
                raise NameError('First molecule found before NUMMOL statement')

            # Create entry into molnames dictionary
            molnames[m.string.split()[1]] = m.string.split()[2] 
            molnamelist.append(m.string.split()[1])  # append to molecule list
            molstart.append(len(sym))
            verboseprint(molnamelist, molstart)
            counter += 1
            continue
        symbol, label = m.group(1, 2)
        symbol = symbol.capitalize()

        if not symbols:
            assert symbol in chemical_symbols, 'Unknown chemical symbol.\
Line is:  {:}'.format(line)
            sym.append(symbol)

        # make sure label is not empty
        if label:
            labels.append(label)
        else:
            labels.append(line.split()[0])

        x, y, z = f.readline().split()[:3]
        positions.append([float(x), float(y), float(z)])
        if levcfg > 0:
            vx, vy, vz = f.readline().split()[:3]
            velocities.append([float(vx), float(vy), float(vz)])
        if levcfg > 1:
            fx, fy, fz = f.readline().split()[:3]
            forces.append([float(fx), float(fy), float(fz)])

    if symbols:
        assert a + 1 == len(symbols), ("Error, counter is at {:} but you gave {:} symbols".format(a + 1, len(symbols)))

    if imcon == 0:
        pbc = False
    elif imcon == 6:
        pbc = [True, True, False]
    else:
        assert imcon in [1, 2, 3]
        pbc = True

    if molformat:  # if/else for returning single Atoms vs dict of Atoms
        atomdict = {}  # dict of Atoms objects to return
        for i in range(len(molnamelist)):  # loop across all detected molecules
            mol_length = int(molnames[molnamelist[i]])  # define number of atoms in selected molecule
            start_index = molstart[i]  # start position in lists (e.g. symbols)
            verboseprint('i', i, 'start_index', start_index)
            atoms = Atoms(positions=positions[start_index:start_index + mol_length],
                          symbols=sym[start_index:start_index + mol_length],
                          cell=cell,
                          pbc=pbc,
                          # Cell is centered around (0, 0, 0) in dlp4:
                          celldisp=-cell.sum(axis=0) / 2
                          )  # atoms object writing
            atoms.set_array(DLP4_LABELS_KEY, labels[start_index:start_index + mol_length], str)  # copied from below
            if levcfg > 0:
                atoms.set_velocities(velocities[start_index:start_index + mol_length])
            if levcfg > 1:
                atoms.set_calculator(SinglePointCalculator(atoms, forces=forces[start_index:start_index + mol_length]))
            atomdict[i] = [molnamelist[i], atoms]  # write to dict (wasn't sure where else to put moelcule name)

        assert len(atomdict) == int(nummol), 'Number of molecules read in incorrectly - {:} vs {:} expected'.format(
            len(atomdict), nummol)
        return atomdict
    else:  # as before
        atoms = Atoms(positions=positions,
                      symbols=sym,
                      cell=cell,
                      pbc=pbc,
                      # Cell is centered around (0, 0, 0) in dlp4:
                      celldisp=-cell.sum(axis=0) / 2)

        atoms.set_array(DLP4_LABELS_KEY, labels, str)
        if levcfg > 0:
            atoms.set_velocities(velocities)
        if levcfg > 1:
            atoms.set_calculator(SinglePointCalculator(atoms, forces=forces))
        return atoms


def read_dlm(f, symbols=None, verbose=False):
    """Read a DL_MONTE config/revcon file.

    Typically used indirectly through read('filename', atoms, format='dlm').

    Can be unforgiving with custom chemical element names.
    Please complain to joseph.manning@manchester.ac.uk in case of bugs.

    :param f: IOWrapper object containing your ASCII DL_MONTE config file
    :param symbols: a list of atomtype symbols, passed to read_simgle_dlm_image
    :param verbose: Bool. Prints useful debugging information if True
    :yield: a generator containing each frame of f
    """
    verboseprint = print if verbose else lambda *a, **k: None
    try:
        verboseprint('Reading in file', f.name)
    except AttributeError:
        verboseprint('Reading in a file with no name')
    f.readline()
    line = f.readline()
    tokens = line.split()
    levcfg = int(tokens[0])
    verboseprint('Positional data only' if levcfg == 0 else 'levcfg = {}'.format(levcfg))
    imcon = int(tokens[1])
    coord_style = {
        0: 'Fractional coordinates used',
        1: 'Cartesian coordinates used',
    }
    verboseprint(coord_style[imcon] if imcon in coord_style.keys() else 'imcon = {}'.format(imcon))

    position = f.tell()

    try:
        is_trajectory = tokens[3] == 'EXTRAS'
        verboseprint('Trajectory file format detected.')
    except IndexError:
        is_trajectory = False
        verboseprint('Single frame detected. Reading to the end of file.')

    if not is_trajectory:
        f.seek(position)

    while line:
        if is_trajectory:
            tokens = line.split()
            natoms = int(tokens[2])
            verboseprint(natoms, 'atoms found to read in.')
        else:
            natoms = None
        yield read_single_dlm_image(f, levcfg, imcon, natoms, 
                                    is_trajectory, symbols, verbose)
        line = f.readline()


DLP4_LABELS_KEY = 'dlp4_labels'


def _get_frame_positions_dlm(f):
    """Get positions of frames in a DL_MONTE HISTORY file."""
    # header line contains name of system
    init_pos = f.tell()
    f.seek(0)
    rl = len(f.readline())  # system name, and record size
    items = f.readline().strip().split()
    print(items)
    if len(items) == 5:
        classic = False
        dlmonte = False
        dlpoly = False
    elif len(items) == 3:
        classic = True
        dlmonte = False
        dlpoly = False
    elif len(items) == 7:
        dlmonte = False
        classic = False
        dlpoly = True
    elif len(items) == 8:
        dlmonte = True
        classic = False
        dlpoly = False
    else:
        raise RuntimeError("Cannot determine version of HISTORY file format.")

    levcfg, imcon, natoms = [int(x) for x in items[0:3]]
    # print(classic,dlpoly, dlmonte)

    if classic or dlpoly:
        # we have to iterate over the entire file
        startpos = f.tell()
        pos = []
        line = True
        while line:
            line = f.readline()
            if 'timestep' in line:
                pos.append(f.tell())
        f.seek(startpos)
    elif dlmonte:
        natoms = []
        # we have to iterate over the entire file
        # print('hello')
        f.seek(0)
        startpos = f.tell()
        # print('startpos', startpos)
        pos = []
        line = True
        while line:
            line = f.readline()
            if 'EXTRAS' in line:
                # print(line.strip().split())
                pos.append(f.tell())
                natoms.append(int(line.strip().split()[2]))
        f.seek(startpos)
    else:
        nframes = int(items[3])
        pos = [((natoms * (levcfg + 2) + 4) * i + 3) * rl for i in range(nframes)]
    f.seek(init_pos)
    return levcfg, imcon, natoms, pos


def read_dlm_history(f, index=-1, symbols=None):
    """Read a DL_MONTE HISTORY file.

    Compatible with DL_MONTE 2.07 (and possibly earlier)

    *Index* can be integer or slice.

    Provide a list of element strings to symbols to ignore naming
    from the HISTORY file.

    :param index:
    :param f: IOWrapper object containing your ASCII DL_MONTE config file
    :param symbols: a list of atomtype symbols passed to read_simgle_dlm_image
    :return: A list of dictionaries with the format {name: Atoms object}
    """
    levcfg, imcon, natoms, pos = _get_frame_positions_dlm(f)
    print(pos)
    if isscalar(index):
        selected = [pos[index]]
        selected_natoms = [natoms[index]]
    else:
        selected = pos[index]
        selected_natoms = natoms[index]

    images = []
    for fpos, fnatoms in zip(selected, selected_natoms):
        print('seeking frame', fpos)
        f.seek(fpos + 1)
        print('fnatoms', fnatoms)
        images.append(read_single_dlm_image(f, levcfg, imcon, fnatoms,
                                            is_trajectory=True, 
                                            symbols=symbols))

    return images


def iread_dlm_history(f, symbols=None):
    """Generator version of read_dlm_history"""
    levcfg, imcon, natoms, pos = _get_frame_positions_dlm(f)
    for p, pnatoms in zip(pos, natoms):
        print('reading frame {0} out of {1}'.format(pos.index(p), len(pos)))
        f.seek(p + 1)
        yield read_single_dlm_image(f, levcfg, imcon, pnatoms,
                                    is_trajectory=True, symbols=symbols)
