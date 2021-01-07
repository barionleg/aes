"""Test module for explicitly unittesting parts of the VASP calculator"""

import pytest

from ase.build import molecule
from ase.calculators.calculator import CalculatorSetupError
from ase.calculators.vasp import Vasp
from ase.calculators.vasp.vasp import check_atoms, check_pbc, check_cell


@pytest.fixture
def atoms():
    _atoms = molecule('H2', vacuum=5)
    _atoms.pbc = True
    return _atoms


def test_check_atoms(atoms, mock_vasp_calculate):
    check_atoms(atoms)
    check_pbc(atoms)
    check_cell(atoms)


@pytest.mark.parametrize(
    'bad_atoms',
    [
        None,
        'a_string',
        # We cannot handle lists of atoms either
        [molecule('H2', vacuum=5)],
    ])
def test_not_atoms(bad_atoms, mock_vasp_calculate):

    with pytest.raises(CalculatorSetupError):
        check_atoms(bad_atoms)

    calc = Vasp()
    with pytest.raises(CalculatorSetupError):
        calc.calculate(atoms=bad_atoms)


@pytest.mark.parametrize('pbc', [
    3 * [False],
    [True, False, True],
    [False, True, False],
])
def test_bad_pbc(atoms, pbc, mock_vasp_calculate):
    """Test handling of PBC"""
    atoms.pbc = pbc

    check_cell(atoms)  # We have a cell, so this should not raise

    # Check that our helper functions raises the expected error
    with pytest.raises(CalculatorSetupError):
        check_pbc(atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(atoms)

    # Check we also raise in the calculator when launching
    # a calculation, but before VASP is actually executed
    calc = Vasp()
    atoms.calc = calc
    with pytest.raises(CalculatorSetupError):
        atoms.get_potential_energy()


def test_vasp_no_cell(mock_vasp_calculate):
    """
    Check VASP input handling.
    """
    # Molecules come with no unit cell
    atoms = molecule('CH4')
    # We should not have a cell
    assert atoms.get_cell().sum() == 0

    with pytest.raises(CalculatorSetupError):
        check_cell(atoms)
    with pytest.raises(CalculatorSetupError):
        check_atoms(atoms)

    with pytest.raises(RuntimeError):
        atoms.write('POSCAR')

    calc = Vasp()
    atoms.calc = calc
    with pytest.raises(CalculatorSetupError):
        atoms.get_total_energy()
