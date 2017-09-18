import numpy as np
import math
from itertools import combinations, permutations, product
from ase.db import connect
from ase.ce.settings import BulkCrystal
from ase.ce.tools import wrap_and_sort_by_position

def index_of_matching_row(A, x):
    for i in range(A.shape[0]):
        if np.allclose(np.array(A[i]), np.array(x)):
            return i
    return None


class CorrFunction(object):
    """
    Class that stores the necessary information about rocksalt structures.
    """
    def __init__(self, BC):
        if type(BC) is not BulkCrystal:
            raise TypeError("Passed object should be BulkCrystal type")
        self._cell_dim = BC.cell_dim
        self.cluster_names = BC.cluster_names
        self.spin_dict = BC.spin_dict
        self.basis_functions = BC.basis_functions
        self.alat = BC.alat
        self.min_cluster_size = BC.min_cluster_size
        self.max_cluster_size = BC.max_cluster_size
        self.max_cluster_dia = BC.max_cluster_dia
        self.dist_matrix = BC.dist_matrix
        self.cluster_names = BC.cluster_names
        self.cluster_dist = BC.cluster_dist
        self.cluster_indx = BC.cluster_indx
        self.trans_matrix = BC.trans_matrix
        self.db = connect(BC.db_name)


    def get_min_distance(self, cluster):
        d = []
        for t in range(8):
            row = []
            for x in combinations(cluster, 2):
                row.append(self.dist_matrix[x[0], x[1], t])
            d.append(sorted(row, reverse=True))
        return np.array(min(d))


    def get_c1(self, atoms, dec):
        c1 = 0
        for element, spin in self.basis_functions[dec].items():
            num_element = len([a for a in atoms if a.symbol == element])
            c1 += num_element*spin
        c1 /= float(len(atoms))
        return c1


    def get_cf(self, atoms):
        atoms = self.check_and_convert_cell_size(atoms.copy())
        natoms = len(atoms)
        bf_list = list(range(len(self.basis_functions)))
        cf = {}
        # ----------------------------------------------------
        # Compute correlation function up the max_cluster_size
        # ----------------------------------------------------
        for n in range(self.min_cluster_size, self.max_cluster_size+1):
            perm = list(product(bf_list, repeat=n))
            if n == 0:
                cf['c0'] = 1.
                continue
            if n == 1:
                for i, dec in enumerate(perm):
                    cf['c1_{}'.format(i+1)] = self.get_c1(atoms, dec[0])
                continue
            indx_list = self.cluster_indx[n]
            name_list = self.cluster_names[n]
            # first category
            for cat in range(len(name_list)):
                for i, dec in enumerate(perm):
                    sp = self.spin_product(atoms, indx_list[cat], dec)
                    count = len(indx_list[cat])*natoms
                    cf_temp = sp/count
                    cf['{}_{}'.format(name_list[cat], i+1)] = cf_temp
        return cf

    def get_cf_by_cluster_names(self, atoms, cluster_names):
        atoms = self.check_and_convert_cell_size(atoms.copy())
        natoms = len(atoms)
        cf = {}
        # ----------------------------------------------------
        # Compute correlation function up the max_cluster_size
        # ----------------------------------------------------
        for name in cluster_names:
            if name == 'c0':
                cf[name] = 1.
                continue
            elif name == 'c1':
                cf[name] = self.get_c1(atoms)
                continue
            # find the location of the name in cluster_names
            for n in range(2, len(self.cluster_names)):
                try:
                    ctype = self.cluster_names[n].index(name)
                    num = n
                    break
                except ValueError:
                    continue
            count = len(self.cluster_indx[num][ctype])*natoms
            sp = self.spin_product(atoms, self.cluster_indx[num][ctype])
            sp /= count
            cf[name] = sp
        return cf

    def spin_product(self, atoms, indx_list, dec):
        num_indx = len(indx_list)
        bf = self.basis_functions
        sp = 0.
        # spin product of each atom from 0 to N
        for atom in atoms:
            ref_indx = atom.index
            ref_spin = bf[dec[0]][atoms[ref_indx].symbol]
            for i in range(num_indx):
                sp_temp = ref_spin
                for j, indx in enumerate(indx_list[i][:]):
                    trans_indx = self.trans_matrix[ref_indx, indx]
                    sp_temp *= bf[dec[j+1]][atoms[trans_indx].symbol]
                sp += sp_temp
        return sp

    def check_and_convert_cell_size(self, atoms):
        """
        Check if the size of the provided cell is the same as the size of the
        template stored in the database. If it either (1) has the same size or
        (2) can make the same size by simple multiplication (supercell), the
        cell with the same size is returned after it is sorted by the position
        and wrapped. If not, it raises an error.
        """
        cell_lengths = atoms.get_cell_lengths_and_angles()[:3]\
                       /(2.*self.alat)*1000
        try:
            row = self.db.get(name='information')
            template = row.toatoms()
        except:
            raise IOError("Cannot retrieve the information template from the"+
                          " database")
        template_lengths = template.get_cell_lengths_and_angles()[:3]\
                           /(2.*self.alat)*1000

        if np.allclose(cell_lengths, template_lengths):
            atoms = wrap_and_sort_by_position(atoms)
        else:
            ratios = template_lengths/cell_lengths
            int_ratios = ratios.round(decimals=0).astype(int)
            if np.allclose(ratios, int_ratios):
                atoms = wrap_and_sort_by_position(atoms*int_ratios)
            else:
                raise TypeError("Cannot make the passed atoms object to the "+
                                "specified size {}".format(self._cell_dim))
        return atoms
