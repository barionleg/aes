"""Definition of SwapAtoms Class.

This module defines the manipulation to Atoms object for Monte Carlo and
Simulated Annealing.
"""
from random import choice
import numpy as np
from ase.ce.settings import BulkCrystal
from ase.atoms import Atoms


class SwapAtoms(object):
    """Class for swapping two atoms in Atoms object with several different
    constraints. To be used for Monte Carlo and Simulated Annealing.
    """
    def __init__(self, setting):
        if not isinstance(setting, BulkCrystal):
            raise TypeError("Passed object should be BulkCrystal type")
        self.setting = setting

    @staticmethod
    def change_element_type(atoms, candidate_indices=None,
                            candidate_symbols=None, rplc_symbols=None):
        """Changes the type of element for an atom in init_atoms object.

        If index and replacing element types are not specified, they are
        randomly generated.

        Arguments
        ==========
        init_atoms: Atoms object

        candidate_indices: (list) possible atomic indices of the atom getting
                           replaced

        candidate_symbols: (list) possible symbols (e.g., ['H', 'C']) of
                           element to be replaced

        rplc_symbols: (list) possible symbols to replace the existing atom with

        Notes
        =====
        1. If candidate_indices = candidate_element = None, pick any atom from
           init_atoms
        2. If candidate_indices is specified (list of indices),
           candidate_symbols is ignored.
        3. If rplc_element = None, it will be randomly chosen from the types of
           elements constituting the passed atoms object (minus the type of
           atom getting replaced). Equal probability of selection for all
           symbols.
        4. If different probability for rplc_symbol is wanted, pass list with
           repeating symbols (e.g., rplc_symbols=['O', 'O', 'C'] will have 2X
           probability of selecting 'O' than selecting 'C'.
        """
        # Check init_atoms
        check_atoms(atoms, at_least_two_element_types=False)

        # Check parameters for candidate_indices and candidate_symbols.
        # Assign default values if missing.
        if candidate_indices is None and candidate_symbols is None:
            candidate_indices = [a.index for a in atoms]
        elif candidate_indices is not None:
            if not isinstance(candidate_indices, list):
                raise TypeError('candadiate_indices should be list type')
        else:
            if not isinstance(candidate_symbols, list):
                raise TypeError('candadiate_symbols should be list type')
            candidate_indices = [a.index for a in atoms if a.symbol in
                                 candidate_symbols]

        indx = choice(candidate_indices)
        symbol = atoms[indx].symbol

        if rplc_symbols is None:
            rplc_symbols = list(set(atoms.get_chemical_symbols()))
        else:
            if not isinstance(rplc_symbols, list):
                raise TypeError('rplc_symbols should be list type')

        if symbol in rplc_symbols:
            rplc_symbols.remove(symbol)
        if not rplc_symbols:
            raise ValueError('resulting rplc_symbols list is empty.')

        atoms[indx].symbol = choice(rplc_symbols)

    @staticmethod
    def swap_nn_atoms(atoms):
        """Swap two nearest neighbor atoms and return indices of the two
        swapped atoms."""
        check_atoms(atoms)
        all_indices = [a.index for a in atoms]
        indx = np.zeros(2, dtype=int)
        symbol = [None] * 2
        nn_indices = []

        # Pick fist atom and determine its symbol and type.
        # First atom should have at least one NN that is different type of
        # element.
        while not nn_indices:
            indx[0] = choice(all_indices)
            symbol[0] = atoms[indx[0]].symbol
            nn_indices = find_nn(atoms, indx[0], all_indices)
            nn_indices = filter_same_element(atoms, symbol[0], nn_indices)

        # Pick second atom that is not the same element
        indx[1] = choice(nn_indices)
        symbol[1] = atoms[indx[1]].symbol

        # Swap two atoms
        atoms[indx[0]].symbol = symbol[1]
        atoms[indx[1]].symbol = symbol[0]
        return indx

    @staticmethod
    def swap_any_two_atoms(atoms):
        """Swap two randomly chosen atoms and return indices of the two
        swapped atoms."""
        check_atoms(atoms)
        all_indices = [a.index for a in atoms]
        indx = np.zeros(2, dtype=int)
        symbol = [None] * 2

        # Pick fist atom and determine its symbol and type
        indx[0] = choice(all_indices)
        symbol[0] = atoms[indx[0]].symbol
        filtered_indices = filter_same_element(atoms, symbol[0], all_indices)
        # Pick second atom that is not the same element
        indx[1] = choice(filtered_indices)
        symbol[1] = atoms[indx[1]].symbol

        # Swap two atoms
        atoms[indx[0]].symbol = symbol[1]
        atoms[indx[1]].symbol = symbol[0]
        return indx

    @staticmethod
    def swap_by_indices(atoms, indx1, indx2):
        """Swap two atoms with the provided indices"""
        # Get symbols
        symbol1 = atoms[indx1].symbol
        symbol2 = atoms[indx2].symbol

        # Swap two atoms
        atoms[indx1].symbol = symbol2
        atoms[indx2].symbol = symbol1

    #
    # def swap_any_two_in_same_basis(self, init_atoms):
    #     return True
    #
    # def swap_NN_in_same_basis(self, init_atoms):
    #     return True


def check_atoms(atoms, at_least_two_element_types=True):
    """Check the validity of passed atoms argument.
    (1) Check if it is Atoms object
    (2) if 'at_least_two_element_types=True,' return True only when it
        contains more than one element type
    """
    if not isinstance(atoms, Atoms):
        raise TypeError("passed argument is not Atoms object")
    # ensure that atoms contain more than 1 type of elements
    if at_least_two_element_types and len(np.unique(atoms.numbers)) < 2:
        raise TypeError("Atoms object needs to have at least two different "
                        "types of elements.")
    return True


def filter_same_element(atoms, symbol, indx_list):
    """Return the indices of elements in atoms that do not have symbol that
    matches the passed atomic symbol.
    """
    s_list = [a.symbol for a in atoms if a.index in indx_list]
    rm_indx = [i for i, j in enumerate(s_list) if j == symbol]
    f_list = [j for i, j in enumerate(indx_list) if i not in rm_indx]
    return f_list


def find_nn(atoms, ref_indx, indx_list):
    """Return indices of atoms that are the nearst neighbors of the reference
    atom (given by ref_indx). It takes a list of indices (indx_list) of
    possible candidate atoms and returns the indices of atoms that has the
    minimum distance from the reference atom.
    """
    indx_list.remove(ref_indx)
    dists = atoms.get_distances(ref_indx, indx_list, mic=True)
    min_dist = min(dists)
    keep_indices = [i for i, j in enumerate(dists) if abs(j - min_dist) < 0.01]
    nn_list = [j for i, j in enumerate(indx_list) if i in keep_indices]
    return nn_list
