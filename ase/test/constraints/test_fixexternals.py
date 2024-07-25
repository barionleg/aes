import numpy as np
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.constraints import FixExternals, FixAtoms
from ase.build import fcc111, add_adsorbate, molecule
from ase.md.verlet import VelocityVerlet
from ase import units
import pytest


def setup_atoms():
    atoms = fcc111(symbol='Cu', size=[3, 3, 4], a=3.58)
    adsorbate = molecule('CH3OH')
    add_adsorbate(atoms, adsorbate, 2.5, 'ontop')
    atoms.center(vacuum=8.5, axis=2)
    return atoms


def setup_list_of_indices():
    indices = []
    indices.append([36, 37, 38, 39, 40, 41])
    indices.append([36, 37, 38, 39, 40])
    indices.append([38, 36, 41, 40, 37])
    indices.append([36, 37, 38, 39])
    indices.append([37, 38, 39])
    indices.append([37, 38])
    indices.append([38, 39])
    return indices


def setup_fixexternals():
    atoms = setup_atoms()
    indices = setup_list_of_indices()
    constraint_list = []
    fix_surface = \
        FixAtoms(indices=[atom.index for atom in atoms if atom.symbol == 'Cu'])
    for i in range(len(indices)):
        tmp_c = FixExternals(atoms, indices[i])
        constraint_list.append(tmp_c)
    return atoms, constraint_list, fix_surface


def displace_atoms_randomly(atoms, c):
    shape = np.shape(atoms[c.indices].positions)
    dx = np.random.rand(shape[0], shape[1])
    displaced_atoms = atoms.copy()
    displaced_atoms.positions[c.indices, :] += dx
    return displaced_atoms


def test_fixexternals_BFGS():
    atoms, constraint_list, fix_surface = setup_fixexternals()
    for i in range(len(constraint_list)):
        tmp_atoms = atoms.copy()
        c = constraint_list[i]
        indices = c.indices
        tmp_atoms.set_constraint([c, fix_surface])
        tmp_atoms.calc = EMT()
        dyn = BFGS(tmp_atoms)
        dyn.run(steps=3)
        inertia_info = \
            tmp_atoms[indices].get_moments_of_inertia(vectors=True)
        final_pa = c.sort_principle_axes(np.transpose(inertia_info[1]))
        final_com = tmp_atoms[indices].get_center_of_mass()
        assert np.max(final_pa - c.principle_axes) == \
            pytest.approx(0, rel=1e-6, abs=1e-6)
        assert np.max(final_com - c.center_of_mass) == \
            pytest.approx(0, rel=1e-6, abs=1e-6)


def test_fixexternals_VelocityVerlet():
    atoms, constraint_list, fix_surface = setup_fixexternals()
    for i in range(len(constraint_list)):
        tmp_atoms = atoms.copy()
        c = constraint_list[i]
        indices = c.indices
        tmp_atoms.set_constraint([c, fix_surface])
        tmp_atoms.calc = EMT()
        dyn = VelocityVerlet(tmp_atoms, timestep=5 * units.fs)
        dyn.run(steps=3)
        inertia_info = \
            tmp_atoms[indices].get_moments_of_inertia(vectors=True)
        final_pa = c.sort_principle_axes(np.transpose(inertia_info[1]))
        final_com = tmp_atoms[indices].get_center_of_mass()
        assert np.max(final_pa - c.principle_axes) == \
            pytest.approx(0, rel=1e-6, abs=1e-6)
        assert np.max(final_com - c.center_of_mass) == \
            pytest.approx(0, rel=1e-6, abs=1e-6)


def test_sort_principle_axes():
    atoms, constraint_list, fix_surface = setup_fixexternals()
    for i in range(len(constraint_list)):
        tmp_atoms = atoms.copy()
        c = constraint_list[i]
        displace_atoms_randomly(tmp_atoms, c)
        inertia_info = tmp_atoms[c.indices].get_moments_of_inertia(vectors=True)
        final_pa = c.sort_principle_axes(np.transpose(inertia_info[1]))
        dot00 = \
            abs(np.dot(np.transpose(final_pa[:, 0]), c.principle_axes[:, 0]))
        dot10 = \
            abs(np.dot(np.transpose(final_pa[:, 1]), c.principle_axes[:, 0]))
        dot20 = \
            abs(np.dot(np.transpose(final_pa[:, 2]), c.principle_axes[:, 0]))
        dot11 = \
            abs(np.dot(np.transpose(final_pa[:, 1]), c.principle_axes[:, 1]))
        dot21 = \
            abs(np.dot(np.transpose(final_pa[:, 2]), c.principle_axes[:, 1]))
        assert dot00 > dot10 and dot00 > dot20
        assert dot11 > dot21


def test_adjust_rotation():
    atoms, constraint_list, fix_surface = setup_fixexternals()
    for i in range(len(constraint_list)):
        tmp_atoms = atoms.copy()
        c = constraint_list[i]
        indices = c.indices
        atoms = displace_atoms_randomly(tmp_atoms, c)
        tmp_atoms.positions[indices, :] = \
            np.copy(c.adjust_rotation(tmp_atoms[indices]))
        inertia_info = tmp_atoms[indices].get_moments_of_inertia(vectors=True)
        final_pa = c.sort_principle_axes(np.transpose(inertia_info[1]))
        final_com = tmp_atoms[indices].get_center_of_mass()
        assert np.max(final_pa - c.principle_axes) == \
            pytest.approx(0, rel=1e-6, abs=1e-6)
        assert np.max(final_com - c.center_of_mass) == \
            pytest.approx(0, rel=1e-6, abs=1e-6)


def test_subspace():
    atoms, constraint_list, fix_surface = setup_fixexternals()
    for j in range(len(constraint_list)):
        tmp_atoms = atoms.copy()
        c = constraint_list[j]
        indices = c.indices
        J_sub = c.get_subspace(tmp_atoms[indices])
        for i in range(np.shape(J_sub)[1]):
            tmpi_atoms = tmp_atoms.copy()
            tmpi_atoms.positions[indices, :] += \
                1e-4 * J_sub[:, i].reshape(-1, 3)
            inertia_info = \
                tmpi_atoms[indices].get_moments_of_inertia(vectors=True)
            final_pa = c.sort_principle_axes(np.transpose(inertia_info[1]))
            final_com = tmpi_atoms[indices].get_center_of_mass()
            assert np.max(final_pa - c.principle_axes) == \
                pytest.approx(0, rel=1e-6, abs=1e-6)
            assert np.max(final_com - c.center_of_mass) == \
                pytest.approx(0, rel=1e-6, abs=1e-6)
