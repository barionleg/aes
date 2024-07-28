import ast
import operator
import simpleeval
import re

from ase import Atoms
import ase.lattice.hexagonal
from ase.neighborlist import NeighborList
from ase.build.tools import stack as stack_pbc
import numpy as np

def graphene_flake(radius, C_C = 1.42, vacuum = None, prune = True):
    """Create a circular graphene flake.

    Creates a finite circular graphene flake in the x-y plane.

    Parameters:

    radius: float
        The radius of the circular flake.

    C_C: float
        The length of C-C bonds. Default: 1.42 Angstrom.

    vacuum: float
        If specified adjust the amount of vacuum in the
        z-direction on each side. Default: 1.75 Angstrom.

    prune: bool
        If True remove dangling atoms on the edges.
        Default: True

    """

    a = C_C * 3**0.5
    size = int(5 * radius)
    atoms = ase.lattice.hexagonal.Graphene(symbol = 'C', latticeconstant={'a':a, 'c':1}, size=(size, size, 1))
    atoms.set_pbc(False)

    center_atom = np.argmin(np.sum((atoms.get_positions() - atoms.get_center_of_mass())**2, axis = 1))
    center_pos = atoms.get_positions()[center_atom]

    atoms_remove = np.sum((atoms.get_positions() - center_pos)**2, axis = 1) > radius**2
    del atoms[atoms_remove]

    if prune:
        cutoff = 1.10 * C_C / 2.
        nl = NeighborList(len(atoms) * [cutoff,], self_interaction = False, bothways = True, skin = 0.)
        nl.update(atoms)
        atoms_remove = len(atoms) * [False,]
        for iatom, _ in enumerate(atoms):
            indices, _ = nl.get_neighbors(iatom)
            if len(indices) == 1:
                atoms_remove[iatom] = True
        del atoms[atoms_remove]

    atoms.set_cell((1, 1, 1))
    if vacuum is not None:
        atoms.center(vacuum)
    else:
        atoms.center(3.5 / 2)

    return atoms

def layered_assembly(notation, blocks, centers = None, vacuum = None):
    """Create a layered assembly.

    Creates a layered assembly in the z-direction.

    Parameters:

    notation: string
         The string-based notation describing the layered assembly.
         For example, the string G/G@1.08 describes a bilayer of
         graphene with twist angle of 1.08. The symbols / and @ are
         used to describe the binary operations of the vertical
         stacking of a layer or layered (sub)structure on another,
         and counterclockwise rotation of a layer or layered
         (sub)structure by some angle about the stacking direction
         (in degrees; assuming the same absolute coordinate system),
         respectively. The grammar is described in detail in
         G. A. Tritsaris et al. arXiv:1910.03413 (2020)

    blocks: dictionary
         Symbols describing the building blocks (case-insesitive).
         Non-alphanumeric characters will be stripped.

    centers: dictionary
         If specified center the building blocks at 2D point,
         or 'COM' to select the center of mass,
         or 'COP' to select center of positions.

    vacuum: float
         If specified adjust the amount of vacuum when centering.
         Default: No vacuum.
    """

    def has_pbc():
        pbc = [np.sum(atoms.get_pbc()[:-1]) > 0 for atoms in blocks.values()]

        return np.all(pbc)

    def Build(m):
        atoms = blocks[m].copy()
        if centers is not None:
            if centers[m] is None:
                center = (0, 0, 0)
            elif centers[m].upper() == 'COM':
                center = atoms.get_center_of_mass()
            elif centers[m].upper() == 'COP':
                center = atoms.get_center_of_positions()

            if len(center) == 2:
                center = (center[0], center[1], 0)
            else:
                center[2] = 0

            atoms.translate(-center)

        return atoms

    def Rotate(atoms, theta):
        if has_pbc():
            raise IncompatibleCellError('Rotating periodic blocks is not supported.')

        atoms.rotate(theta, 'z')

        return atoms

    def Stack(atoms1, atoms2):
        if has_pbc():
            atoms = stack_pbc(atoms1, atoms2, maxstrain = 1)
        else:
            c1, c2 = atoms1.get_cell(), atoms2.get_cell()
            z1, z2 = c1[2][2], c2[2][2]
            atoms2.positions += (0, 0, z1)
            atoms = atoms1 + atoms2
            atoms.set_cell((0, 0, z1 + z2))

        return atoms

    #sanitize input parameters
    notation = notation.upper()
    for symbol, atoms in blocks.copy().items():
        atoms = Atoms(atoms)

        del blocks[symbol]
        symbol = re.sub(r'\W+', '', symbol)
        symbol = symbol.upper()
        blocks[symbol] = atoms

    NAMES = {}
    for symbol in blocks:
        NAMES[symbol] = 'Build("%s")' % symbol


    OPERATORS = {
        ast.Mult: lambda a,r: f'Rotate({a}, {r})', #higher precedence, use parenthesis to override
        ast.Add: lambda a,b: f'Stack({a}, {b})',
        ast.USub: operator.neg
    }

    FUNCTIONS = {}
    s = simpleeval.EvalWithCompoundTypes(names = NAMES, operators = OPERATORS, functions = FUNCTIONS)

    notation = notation.replace('@', '*').replace('/', '+')
    operations = s.eval(notation)
    atoms = eval(operations, {'Build':Build, 'Rotate':Rotate, 'Stack':Stack})

    if vacuum is None:
        vacuum = 0.0
    if has_pbc():
        atoms.center(vacuum, axis = 2)
    else:
        atoms.center(vacuum)

    return atoms

class IncompatibleCellError(ValueError):
    """Exception raised if rotating fails due to periodic cell"""
    pass