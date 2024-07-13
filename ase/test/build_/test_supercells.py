import itertools

import numpy as np
import pytest

from ase.build import bulk
from ase.build.supercells import (
    make_supercell,
    get_deviation_from_optimal_cell_shape,
    find_optimal_cell_shape,
)


@pytest.fixture()
def rng():
    return np.random.RandomState(seed=42)


@pytest.fixture(
    params=[
        bulk("NaCl", crystalstructure="rocksalt", a=4.0),
        bulk("NaCl", crystalstructure="rocksalt", a=4.0, cubic=True),
        bulk("Au", crystalstructure="fcc", a=4.0),
    ]
)
def prim(request):
    return request.param


@pytest.fixture(
    params=[
        3 * np.diag([1, 1, 1]),
        4 * np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        3 * np.diag([1, 2, 1]),
    ]
)
def P(request):
    return request.param


@pytest.fixture(params=["cell-major", "atom-major"])
def order(request):
    return request.param


def test_make_supercell(prim, P, order):
    n = int(round(np.linalg.det(P)))
    expected = n * len(prim)
    sc = make_supercell(prim, P, order=order)
    assert len(sc) == expected
    if order == "cell-major":
        symbols_expected = list(prim.symbols) * n
    elif order == "atom-major":
        symbols_expected = [s for s in prim.symbols for _ in range(n)]
    assert list(sc.symbols) == symbols_expected


def test_make_supercells_arrays(prim, P, order, rng):
    reps = int(round(np.linalg.det(P)))
    tags = list(range(len(prim)))
    momenta = rng.random((len(prim), 3))

    prim.set_tags(tags)
    prim.set_momenta(momenta)

    sc = make_supercell(prim, P, order=order)

    assert reps * len(prim) == len(sc.get_tags())
    if order == "cell-major":
        assert all(sc.get_tags() == np.tile(tags, reps))
        assert np.allclose(sc[: len(prim)].get_momenta(), prim.get_momenta())
        assert np.allclose(sc.get_momenta(), np.tile(momenta, (reps, 1)))
    elif order == "atom-major":
        assert all(sc.get_tags() == np.repeat(tags, reps))
        assert np.allclose(sc[::reps].get_momenta(), prim.get_momenta())
        assert np.allclose(sc.get_momenta(), np.repeat(momenta, reps, axis=0))


@pytest.mark.parametrize(
    "rep",
    [
        (1, 1, 1),
        (1, 2, 1),
        (4, 5, 6),
        (40, 19, 42),
    ],
)
def test_make_supercell_vs_repeat(prim, rep):
    P = np.diag(rep)

    at1 = prim * rep
    at1.wrap()
    at2 = make_supercell(prim, P, wrap=True)

    assert np.allclose(at1.positions, at2.positions)
    assert all(at1.symbols == at2.symbols)

    at1 = prim * rep
    at2 = make_supercell(prim, P, wrap=False)
    assert np.allclose(at1.positions, at2.positions)
    assert all(at1.symbols == at2.symbols)


def test_get_deviation_from_optimal_cell_shape():
    # also tested via the docs data examples
    # test perfect scores for SC, where cell vector permutation or magnitude
    # do not matter:
    cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for perm, factor in zip(itertools.combinations(range(3), 3), range(1, 9)):
        permuted_cell = [cell[i] * factor for i in perm]
        assert np.isclose(
            get_deviation_from_optimal_cell_shape(permuted_cell, target_shape="sc"), 0.0
        )

    # likewise for FCC:
    cell = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    for perm, factor in zip(itertools.combinations(range(3), 3), range(1, 9)):
        permuted_cell = [cell[i] * factor for i in perm]
        assert np.isclose(
            get_deviation_from_optimal_cell_shape(permuted_cell, target_shape="fcc"),
            0.0,
        )

    # spot check some cases:
    cell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]])
    assert np.isclose(get_deviation_from_optimal_cell_shape(cell, "sc"), 0.6558650332)

    # fcc
    cell = np.array([[0, 1, 1], [1, 0, 1], [2, 2, 0]])
    assert np.isclose(get_deviation_from_optimal_cell_shape(cell, "fcc"), 0.6558650332)

    # negative determinant
    cell = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    assert np.isclose(get_deviation_from_optimal_cell_shape(cell, "sc"), 0.0)


def test_find_optimal_cell_shape():
    # also tested via the docs data examples
    cell = np.diag([1.0, 2.0, 4.0])
    target_size = 8
    target_shape = "sc"
    result = find_optimal_cell_shape(cell, target_size, target_shape)
    assert np.isclose(
        get_deviation_from_optimal_cell_shape(np.dot(result, cell), "sc"), 0.0
    )
    assert np.allclose(np.linalg.norm(np.dot(result, cell), axis=1), 4)

    # docs example:
    conf = bulk("Au")  # fcc
    P1 = find_optimal_cell_shape(conf.cell, 32, "sc")
    assert np.allclose(P1, np.array([[-2, 2, 2], [2, -2, 2], [2, 2, -2]]))

    P1 = find_optimal_cell_shape(conf.cell, 495, "sc")
    assert np.allclose(P1, np.array([[-6, 5, 5], [5, -6, 5], [5, 5, -5]]))
