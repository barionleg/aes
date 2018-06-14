"""Test to initiatialize CE using a BulkSpacegroup.

1. Initialize the CE
2. Add a few structures
3. Compute the energy
4. Run the evaluation routine
"""
import os
from ase.ce import BulkSpacegroup, GenerateStructures, CorrFunction
from ase.db import connect

def test_spgroup_217():
    """Test the initialization of spacegroup 217."""
    a = 10.553
    b = 10.553
    c = 10.553
    alpha = 90
    beta = 90
    gamma = 90
    cellpar = [a, b, c, alpha, beta, gamma]
    basis = [(0, 0, 0), (0.324, 0.324, 0.324),
             (0.3582, 0.3582, 0.0393), (0.0954, 0.0954, 0.2725)]
    conc_args = {"conc_ratio_min_1": [[1, 0]],
                 "conc_ratio_max_1": [[0, 1]]}
    db_name = "test.db"
    basis_elements = [["Al", "Mg"], ["Al", "Mg"], ["Al", "Mg"], ["Al", "Mg"]]

    # Test with grouped basis
    bsg = BulkSpacegroup(basis_elements=basis_elements,
                         basis=basis,
                         spacegroup=217,
                         cellpar=cellpar,
                         conc_args=conc_args,
                         max_cluster_size=4,
                         db_name=db_name,
                         size=[1, 1, 1],
                         grouped_basis=[[0, 1, 2, 3]],
                         max_cluster_dist=5.0)
    os.remove(db_name)

    conc_args = {"conc_ratio_min_1": [[1, 0], [1, 0], [1, 0], [1, 0]],
                 "conc_ratio_max_1": [[0, 1], [0, 1], [0, 1], [0, 1]]}
    basis_elements = [["Al", "Mg"], ["Si", "Mg"], ["Cu", "Mg"], ["Zn", "Mg"]]
    # Test without grouped basis
    bsg = BulkSpacegroup(basis_elements=basis_elements,
                         basis=basis,
                         spacegroup=217,
                         cellpar=cellpar,
                         conc_args=conc_args,
                         max_cluster_size=4,
                         db_name=db_name,
                         size=[1, 1, 1],
                         max_cluster_dist=5.0)
    os.remove(db_name)


def test_grouped_basis_with_large_dist():
    # Test with grouped basis with a supercell
    db_name = "test.db"

    # ---------------------------------- #
    # 2 grouped_basis                    #
    # ---------------------------------- #
    bsg = BulkSpacegroup(basis_elements=[['Li', 'X', 'V'], ['Li', 'X', 'V'],
                                         ['O', 'F']],
                         basis=[(0.00, 0.00, 0.00),
                                (1./3, 2./3, 0.00),
                                (1./3, 0.00, 0.25)],
                         spacegroup=167,
                         cellpar=[5.123, 5.123, 13.005, 90., 90., 120.],
                         size=[1, 1, 1],
                         conc_args={"conc_ratio_min_1": [[0, 2, 1], [2, 1]],
                                    "conc_ratio_max_1": [[2, 0, 1], [2, 1]]},
                         db_name=db_name,
                         grouped_basis=[[0, 1], [2]],
                         max_cluster_size=2,
                         max_cluster_dist=5.0)
    assert bsg.unique_elements == ['F', 'Li', 'O', 'V', 'X']
    assert bsg.spin_dict == {'F': 2.0, 'Li': -2.0, 'O': 1.0, 'V': -1.0, 'X': 0}
    assert len(bsg.basis_functions) == 4
    assert bsg.num_grouped_basis == 2
    assert len(bsg.index_by_grouped_basis) == 2
    assert bsg.num_grouped_elements == 5
    assert len(bsg.basis_functions) == 4
    flat = [i for sub in bsg.index_by_grouped_basis for i in sub]
    background = [a.index for a in bsg.atoms_with_given_dim if
                  a.symbol in bsg.background_symbol]
    assert len(flat) == len(bsg.atoms_with_given_dim) - len(background)

    os.remove(db_name)

    # ---------------------------------- #
    # 2 grouped_basis                    #
    # ---------------------------------- #
    # initial_pool + probe_structures    #
    # ---------------------------------- #
    bsg = BulkSpacegroup(basis_elements=[['O', 'X'], ['O', 'X'],
                                         ['O', 'X'], ['Ta']],
                         basis=[(0., 0., 0.),
                                (0.3894, 0.1405, 0.),
                                (0.201,  0.3461, 0.5),
                                (0.2244, 0.3821, 0.)],
                         spacegroup=55,
                         cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                         size=[1, 2, 2],
                         conc_args={"conc_ratio_min_1": [[5, 0], [2]],
                                    "conc_ratio_max_1": [[4, 1], [2]]},
                         db_name=db_name,
                         max_cluster_size=2,
                         max_cluster_dist=5.0,
                         grouped_basis=[[0, 1, 2], [3]])
    assert bsg.unique_elements == ['O', 'Ta', 'X']
    assert bsg.spin_dict == {'O': 1.0, 'Ta': -1.0, 'X': 0.0}
    assert len(bsg.basis_functions) == 2
    assert bsg.num_grouped_basis == 2
    assert len(bsg.index_by_grouped_basis) == 2
    assert bsg.num_grouped_elements == 3
    assert len(bsg.basis_functions) == 2
    flat = [i for sub in bsg.index_by_grouped_basis for i in sub]
    background = [a.index for a in bsg.atoms_with_given_dim if
                  a.symbol in bsg.background_symbol]
    assert len(flat) == len(bsg.atoms_with_given_dim) - len(background)

    gs = GenerateStructures(setting=bsg, struct_per_gen=3)
    gs.generate_initial_pool()
    gs = GenerateStructures(setting=bsg, struct_per_gen=2)
    gs.generate_probe_structure(init_temp=1.0, final_temp=0.001, num_temp=5,
                                num_steps=10, approx_mean_var=True)
    corrfunc = CorrFunction(setting=bsg)
    db = connect(db_name)
    for row in db.select('id>4'):
        atoms = row.toatoms(add_additional_information=True)
        kvp = atoms.info['key_value_pairs']
        cf = corrfunc.get_cf(atoms, return_type='dict')
        for key, value in cf.items():
            assert kvp[key] - value < 1E-6
    os.remove(db_name)

    # ---------------------------------- #
    # 2 grouped_basis + background atoms #
    # ---------------------------------- #
    # initial_pool + probe_structures    #
    # ---------------------------------- #

    bsg = BulkSpacegroup(basis_elements=[['O', 'X'], ['Ta'], ['O', 'X'],
                                         ['O', 'X']],
                         basis=[(0., 0., 0.),
                                (0.2244, 0.3821, 0.),
                                (0.3894, 0.1405, 0.),
                                (0.201,  0.3461, 0.5)],
                         spacegroup=55,
                         cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                         size=[2, 2, 2],
                         conc_args={"conc_ratio_min_1": [[2], [5, 0]],
                                    "conc_ratio_max_1": [[2], [4, 1]]},
                         db_name=db_name,
                         max_cluster_size=2,
                         max_cluster_dist=5.0,
                         grouped_basis=[[1], [0, 2, 3]],
                         ignore_background_atoms=True)
    assert bsg.unique_elements == ['O', 'X']
    assert bsg.spin_dict == {'O': 1.0, 'X': -1.0}
    assert bsg.basis_elements == [['O', 'X'], ['O', 'X'], ['O', 'X']]
    assert len(bsg.basis_functions) == 1
    assert bsg.num_grouped_basis == 1
    assert len(bsg.index_by_grouped_basis) == 1
    assert bsg.num_grouped_elements == 2
    assert len(bsg.basis_functions) == 1
    flat = [i for sub in bsg.index_by_grouped_basis for i in sub]
    background = [a.index for a in bsg.atoms_with_given_dim if
                  a.symbol in bsg.background_symbol]
    assert len(flat) == len(bsg.atoms_with_given_dim) - len(background)

    gs = GenerateStructures(setting=bsg, struct_per_gen=3)
    gs.generate_initial_pool()
    gs = GenerateStructures(setting=bsg, struct_per_gen=2)
    gs.generate_probe_structure(init_temp=1.0, final_temp=0.001, num_temp=5,
                                num_steps=10, approx_mean_var=True)
    corrfunc = CorrFunction(setting=bsg)
    db = connect(db_name)
    for row in db.select('id>4'):
        atoms = row.toatoms(add_additional_information=True)
        kvp = atoms.info['key_value_pairs']
        cf = corrfunc.get_cf(atoms, return_type='dict')
        for key, value in cf.items():
            assert kvp[key] - value < 1E-6

    os.remove(db_name)


test_spgroup_217()
test_grouped_basis_with_large_dist()
