"""Quantum ESPRESSO file parsers.

Implemented:
* Input file (pwi)
* Output file (pwo) with vc-relax

"""

import io

import numpy as np
import pytest

import ase.build
import ase.io
from ase import Atoms
from ase.calculators.calculator import compare_atoms
from ase.constraints import FixAtoms, FixCartesian, FixScaled
from ase.io.espresso import (get_atomic_species, parse_position_line,
                             read_espresso_in, read_fortran_namelist,
                             write_espresso_in, write_fortran_namelist)

# This file is parsed correctly by pw.x, even though things are
# scattered all over the place with some namelist edge cases
pw_input_text = """
&CONTrol
   prefix           = 'surf_110_H2_md'
   calculation      = 'md'
   restart_mode     = 'from_scratch'
   pseudo_dir       = '.'
   outdir           = './surf_110_!H2_m=d_sc,ratch/'
   verbosity        = 'default'
   tprnfor          = .true.
   tstress          = .True.
!   disk_io          = 'low'
   wf_collect       = .false.
   max_seconds      = 82800
   forc_con!v_thr    = 1e-05
   etot_conv_thr    = 1e-06
   dt               = 41.3 , /

&SYSTEM ecutwfc     = 63,   ecutrho   = 577,  ibrav    = 0,
nat              = 8,   ntyp             = 2,  occupations      = 'smearing',
smearing         = 'marzari-vanderbilt',
degauss          = 0.01,   nspin            = 2,  !  nosym     = .true. ,
    starting_magnetization(2) = 5.12 /
&ELECTRONS
   electron_maxstep = 300
   mixing_beta      = 0.1
   conv_thr         = 1d-07
   mixing_mode      = 'local-TF'
   scf_must_converge = False
/
&IONS
   ion_dynamics     = 'verlet'
   ion_temperature  = 'rescaling'
   tolp             = 50.0
   tempw            = 500.0
/

ATOMIC_SPECIES
H 1.008 H.pbe-rrkjus_psl.0.1.UPF
Fe 55.845 Fe.pbe-spn-rrkjus_psl.0.2.1.UPF

K_POINTS automatic
2 2 2  1 1 1

CELL_PARAMETERS angstrom
5.6672000000000002 0.0000000000000000 0.0000000000000000
0.0000000000000000 8.0146311006808038 0.0000000000000000
0.0000000000000000 0.0000000000000000 27.0219466510212101

ATOMIC_POSITIONS angstrom
Fe 0.0000000000 0.0000000000 0.0000000000 0 0 0
Fe 1.4168000000 2.0036577752 -0.0000000000 0 0 0
Fe 0.0000000000 2.0036577752 2.0036577752 0 0 0
Fe 1.4168000000 0.0000000000 2.0036577752 0 0 0
Fe 0.0000000000 0.0000000000 4.0073155503
Fe 1.4168000000 2.0036577752 4.0073155503
H 0.0000000000 2.0036577752 6.0109733255
H 1.4168000000 0.0000000000 6.0109733255
"""

# Trimmed to only include lines of relevance
pw_output_text = """

     Program PWSCF v.5.3.0 (svn rev. 11974) starts on 19May2016 at  7:48:12

     This program is part of the open-source Quantum ESPRESSO suite
     for quantum simulation of materials; please cite
         "P. Giannozzi et al., J. Phys.:Condens. Matter 21 395502 (2009);
          URL http://www.quantum-espresso.org",
     in publications or presentations arising from this work. More details at
     http://www.quantum-espresso.org/quote

...

     bravais-lattice index     =            0
     lattice parameter (alat)  =       5.3555  a.u.
     unit-cell volume          =     155.1378 (a.u.)^3
     number of atoms/cell      =            3
     number of atomic types    =            2
     number of electrons       =        33.00
     number of Kohn-Sham states=           21
     kinetic-energy cutoff     =     144.0000  Ry
     charge density cutoff     =    1728.0000  Ry
     convergence threshold     =      1.0E-10
     mixing beta               =       0.1000
     number of iterations used =            8  plain     mixing
     Exchange-correlation      = PBE ( 1  4  3  4 0 0)
     nstep                     =           50


     celldm(1)=   5.355484  celldm(2)=   0.000000  celldm(3)=   0.000000
     celldm(4)=   0.000000  celldm(5)=   0.000000  celldm(6)=   0.000000

     crystal axes: (cart. coord. in units of alat)
               a(1) = (   1.000000   0.000000   0.000000 )
               a(2) = (   0.000000   1.010000   0.000000 )
               a(3) = (   0.000000   0.000000   1.000000 )

...

   Cartesian axes

     site n.     atom                  positions (alat units)
         1           Fe  tau(   1) = (   0.0000000   0.0000000   0.0000000  )
         2           Fe  tau(   2) = (   0.5000000   0.5050000   0.5000000  )
         3           H   tau(   3) = (   0.5000000   0.5050000   0.0000000  )

...

     Magnetic moment per site:
     atom:    1    charge:   10.9188    magn:    1.9476    constr:    0.0000
     atom:    2    charge:   10.9402    magn:    1.5782    constr:    0.0000
     atom:    3    charge:    0.8835    magn:   -0.0005    constr:    0.0000

     total cpu time spent up to now is      125.3 secs

     End of self-consistent calculation

     Number of k-points >= 100: set verbosity='high' to print the bands.

     the Fermi energy is    19.3154 ev

!    total energy              =    -509.83425823 Ry
     Harris-Foulkes estimate   =    -509.83425698 Ry
     estimated scf accuracy    <          8.1E-11 Ry

     The total energy is the sum of the following terms:

     one-electron contribution =    -218.72329117 Ry
     hartree contribution      =     130.90381466 Ry
     xc contribution           =     -70.71031046 Ry
     ewald contribution        =    -351.30448923 Ry
     smearing contrib. (-TS)   =       0.00001797 Ry

     total magnetization       =     4.60 Bohr mag/cell
     absolute magnetization    =     4.80 Bohr mag/cell

     convergence has been achieved in  23 iterations

     negative rho (up, down):  0.000E+00 3.221E-05

     Forces acting on atoms (Ry/au):

     atom    1 type  2   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  2   force =     0.00000000    0.00000000    0.00000000
     atom    3 type  1   force =     0.00000000    0.00000000    0.00000000

     Total force =     0.000000     Total SCF correction =     0.000000


     entering subroutine stress ...


     negative rho (up, down):  0.000E+00 3.221E-05
          total   stress  (Ry/bohr**3)                   (kbar)     P=  384.59
   0.00125485   0.00000000   0.00000000        184.59      0.00      0.00
   0.00000000   0.00115848   0.00000000          0.00    170.42      0.00
   0.00000000   0.00000000   0.00542982          0.00      0.00    798.75


     BFGS Geometry Optimization

     number of scf cycles    =   1
     number of bfgs steps    =   0

     enthalpy new            =    -509.8342582307 Ry

     new trust radius        =       0.0721468508 bohr
     new conv_thr            =            1.0E-10 Ry

     new unit-cell volume =    159.63086 a.u.^3 (    23.65485 Ang^3 )

CELL_PARAMETERS (angstrom)
   2.834000000   0.000000000   0.000000000
   0.000000000   2.945239106   0.000000000
   0.000000000   0.000000000   2.834000000

ATOMIC_POSITIONS (angstrom)
Fe       0.000000000   0.000000000   0.000000000    0   0   0
Fe       1.417000000   1.472619553   1.417000000
H        1.417000000   1.472619553   0.000000000


...

     Magnetic moment per site:
     atom:    1    charge:   10.9991    magn:    2.0016    constr:    0.0000
     atom:    2    charge:   11.0222    magn:    1.5951    constr:    0.0000
     atom:    3    charge:    0.8937    magn:   -0.0008    constr:    0.0000

     total cpu time spent up to now is      261.2 secs

     End of self-consistent calculation

     Number of k-points >= 100: set verbosity='high' to print the bands.

     the Fermi energy is    18.6627 ev

!    total energy              =    -509.83806077 Ry
     Harris-Foulkes estimate   =    -509.83805972 Ry
     estimated scf accuracy    <          1.3E-11 Ry

     The total energy is the sum of the following terms:

     one-electron contribution =    -224.15358901 Ry
     hartree contribution      =     132.85863781 Ry
     xc contribution           =     -70.66684834 Ry
     ewald contribution        =    -347.87622740 Ry
     smearing contrib. (-TS)   =      -0.00003383 Ry

     total magnetization       =     4.66 Bohr mag/cell
     absolute magnetization    =     4.86 Bohr mag/cell

     convergence has been achieved in  23 iterations

     negative rho (up, down):  0.000E+00 3.540E-05

     Forces acting on atoms (Ry/au):

     atom    1 type  2   force =     0.00000000    0.00000000    0.00000000
     atom    2 type  2   force =     0.00000000    0.00000000    0.00000000
     atom    3 type  1   force =     0.00000000    0.00000000    0.00000000

     Total force =     0.000000     Total SCF correction =     0.000000


     entering subroutine stress ...


     negative rho (up, down):  0.000E+00 3.540E-05
          total   stress  (Ry/bohr**3)                   (kbar)     P=  311.25
   0.00088081   0.00000000   0.00000000        129.57      0.00      0.00
   0.00000000   0.00055559   0.00000000          0.00     81.73      0.00
   0.00000000   0.00000000   0.00491106          0.00      0.00    722.44


     number of scf cycles    =   2
     number of bfgs steps    =   1

...

Begin final coordinates

CELL_PARAMETERS (angstrom)
   2.834000000   0.000000000   0.000000000
   0.000000000   2.945239106   0.000000000
   0.000000000   0.000000000   2.834000000

ATOMIC_POSITIONS (angstrom)
Fe       0.000000000   0.000000000   0.000000000    0   0   0
Fe       1.417000000   1.472619553   1.417000000
H        1.417000000   1.472619553   0.000000000
End final coordinates

"""


def test_pw_input():
    """Read pw input file."""
    with open('pw_input.pwi', 'w') as pw_input_f:
        pw_input_f.write(pw_input_text)

    pw_input_atoms = ase.io.read('pw_input.pwi', format='espresso-in')
    assert len(pw_input_atoms) == 8
    assert (pw_input_atoms.get_initial_magnetic_moments()
            == pytest.approx([5.12, 5.12, 5.12, 5.12, 5.12, 5.12, 0., 0.]))


def test_get_atomic_species():
    """Parser for atomic species section"""

    with open('pw_input.pwi', 'w') as pw_input_f:
        pw_input_f.write(pw_input_text)
    with open('pw_input.pwi') as pw_input_f:
        data, card_lines = read_fortran_namelist(pw_input_f)
        species_card = get_atomic_species(card_lines,
                                          n_species=data['system']['ntyp'])

    assert len(species_card) == 2
    assert species_card[0] == (
        "H", pytest.approx(1.008), "H.pbe-rrkjus_psl.0.1.UPF")
    assert species_card[1] == (
        "Fe", pytest.approx(55.845), "Fe.pbe-spn-rrkjus_psl.0.2.1.UPF")


def test_pw_output():
    """Read pw output file."""
    with open('pw_output.pwo', 'w') as pw_output_f:
        pw_output_f.write(pw_output_text)

    pw_output_traj = ase.io.read('pw_output.pwo', index=':')
    assert len(pw_output_traj) == 2
    assert pw_output_traj[1].get_volume() > pw_output_traj[0].get_volume()


def test_pw_parse_line():
    """Parse a single position line from a pw.x output file."""
    txt = """       994           Pt  tau( 994) = \
(   1.4749849   0.7329881   0.0719387  )
       995           Sb  tau( 995) = (   1.4212023   0.7037863   0.1242640  )
       996           Sb  tau( 996) = (   1.5430640   0.7699524   0.1700400  )
       997           Sb  tau( 997) = (   1.4892815   0.7407506   0.2223653  )
       998           Sb  tau( 998) = (   1.6111432   0.8069166   0.2681414  )
       999           Sb  tau( 999) = (   1.5573606   0.7777148   0.3204667  )
      1000           Sb  tau(1000) = (   1.6792223   0.8438809   0.3662427  )
      1001           Sb  tau(1001) = (   1.6254398   0.8146791   0.4185680  )
      1002           Sb  tau(1002) = (   1.7473015   0.8808452   0.4643440  )
      1003           Sb  tau(1003) = (   1.6935189   0.8516434   0.5166693  )
"""
    x_result = [1.4749849, 1.4212023, 1.5430640, 1.4892815, 1.6111432,
                1.5573606, 1.6792223, 1.6254398, 1.7473015, 1.6935189]
    y_result = [0.7329881, 0.7037863, 0.7699524, 0.7407506, 0.8069166,
                0.7777148, 0.8438809, 0.8146791, 0.8808452, 0.8516434]
    z_result = [0.0719387, 0.1242640, 0.1700400, 0.2223653, 0.2681414,
                0.3204667, 0.3662427, 0.4185680, 0.4643440, 0.5166693]

    for i, line in enumerate(txt.splitlines()):
        sym, x, y, z = parse_position_line(line)
        if i == 0:
            assert sym == "Pt"
        else:
            assert sym == "Sb"
        assert abs(x - x_result[i]) < 1e-7
        assert abs(y - y_result[i]) < 1e-7
        assert abs(z - z_result[i]) < 1e-7


def test_pw_results_required():
    """Check only configurations with results are read unless requested."""
    with open('pw_output.pwo', 'w') as pw_output_f:
        pw_output_f.write(pw_output_text)

    # ignore 'final coordinates' with no results
    pw_output_traj = ase.io.read('pw_output.pwo', index=':')
    assert 'energy' in pw_output_traj[-1].calc.results
    assert len(pw_output_traj) == 2
    # include un-calculated final config
    pw_output_traj = ase.io.read('pw_output.pwo', index=':',
                                 results_required=False)
    assert len(pw_output_traj) == 3
    assert 'energy' not in pw_output_traj[-1].calc.results
    # get default index=-1 with results
    pw_output_config = ase.io.read('pw_output.pwo')
    assert 'energy' in pw_output_config.calc.results
    # get default index=-1 with no results "final coordinates'
    pw_output_config = ase.io.read('pw_output.pwo', results_required=False)
    assert 'energy' not in pw_output_config.calc.results


def test_pw_input_write():
    """Write a structure and read it back."""
    bulk = ase.build.bulk('NiO', 'rocksalt', 4.813, cubic=True)
    bulk.set_initial_magnetic_moments([2.2 if atom.symbol == 'Ni' else 0.0
                                       for atom in bulk])

    fh = 'espresso_test.pwi'
    pseudos = {'Ni': 'potato', 'O': 'orange'}

    write_espresso_in(fh, bulk, pseudopotentials=pseudos)
    readback = read_espresso_in('espresso_test.pwi')
    assert np.allclose(bulk.positions, readback.positions)

    sections = {'system': {
        'lda_plus_u': True,
        'Hubbard_U(1)': 4.0,
        'Hubbard_U(2)': 0.0}}
    write_espresso_in(fh, bulk, sections, pseudopotentials=pseudos)
    readback = read_espresso_in('espresso_test.pwi')
    assert np.allclose(bulk.positions, readback.positions)


def test_pw_input_write_nested_flat():
    """Write a structure and read it back."""
    bulk = ase.build.bulk('Fe')

    fh = 'espresso_test.pwi'
    pseudos = {'Fe': 'carrot'}

    input_data = {"control": {"calculation": "scf"},
                  "unused_keyword1": "unused_value1",
                  "used_sections": {"used_keyword1": "used_value1"}
                  }

    with pytest.raises(DeprecationWarning):
        write_espresso_in(fh, bulk, input_data=input_data,
                          pseudopotentials=pseudos,
                          mixing_mode="local-TF")

    write_espresso_in(fh, bulk, input_data=input_data,
                      pseudopotentials=pseudos,
                      unusedkwarg="unused")

    with open(fh) as f:
        new_atoms = read_espresso_in(f)
        f.seek(0)
        readback = read_fortran_namelist(f)

    read_string = readback[0].to_string()

    assert "&USED_SECTIONS\n" in read_string
    assert "   used_keyword1    = 'used_value1'\n" in read_string
    assert np.allclose(bulk.positions, new_atoms.positions)


def test_write_fortran_namelist_pw():
    fd = io.StringIO()
    input_data = {
        "calculation": "scf",
        "ecutwfc": 30.0,
        "ibrav": 0,
        "nat": 10,
        "nbnd": 8,
        "conv_thr": 1e-6,
        "random": True}
    binary = "pw"
    write_fortran_namelist(fd, input_data, binary)
    result = fd.getvalue()
    assert "scf" in result
    assert "ibrav" in result
    assert "conv_thr" in result
    assert result.endswith("EOF")
    fd.seek(0)
    reread = read_fortran_namelist(fd)
    assert reread != input_data


def test_write_fortran_namelist_fields():
    fd = io.StringIO()
    input_data = {
        "INPUT": {
            "amass": 28.0855,
            "niter_ph": 50,
            "tr2_ph": 1e-6,
            "flfrc": "silicon.fc"},
    }
    binary = "q2r"
    write_fortran_namelist(
        fd,
        input_data,
        binary,
        additional_cards="test1\ntest2\ntest3\n")
    result = fd.getvalue()
    expected = ("&INPUT\n"
                "   flfrc            = 'silicon.fc'\n"
                "   amass            = 28.0855\n"
                "   niter_ph         = 50\n"
                "   tr2_ph           = 1e-06\n"
                "/\n"
                "test1\n"
                "test2\n"
                "test3\n"
                "EOF")
    assert result == expected


def test_write_fortran_namelist_list_fields():
    fd = io.StringIO()
    input_data = {
        "PRESS_AI": {
            "amass": 28.0855,
            "niter_ph": 50,
            "tr2_ph": 1e-6,
            "flfrc": "silicon.fc"},
    }
    binary = "cp"
    write_fortran_namelist(
        fd,
        input_data,
        binary,
        additional_cards=[
            "test1",
            "test2",
            "test3"])
    result = fd.getvalue()
    expected = ("&CONTROL\n"
                "/\n"
                "&SYSTEM\n"
                "/\n"
                "&ELECTRONS\n"
                "/\n"
                "&IONS\n"
                "/\n"
                "&CELL\n"
                "/\n"
                "&PRESS_AI\n"
                "   amass            = 28.0855\n"
                "   niter_ph         = 50\n"
                "   tr2_ph           = 1e-06\n"
                "   flfrc            = 'silicon.fc'\n"
                "/\n"
                "&WANNIER\n"
                "/\n"
                "test1\n"
                "test2\n"
                "test3\n"
                "EOF")
    assert result == expected


class TestConstraints:
    """Test if the constraint can be recovered when writing and reading.

    Notes
    -----
    Linear constraints in the ATOMIC_POSITIONS block in the quantum ESPRESSO
    `.pwi` format apply to Cartesian coordinates, regardless of whether the
    atomic positions are written in the "angstrom" or the "crystal" units.
    """

    # TODO: test also mask for FixCartesian

    @staticmethod
    def _make_atoms_ref():
        """water molecule"""
        atoms = ase.build.molecule("H2O")
        atoms.cell = 10.0 * np.eye(3)
        atoms.pbc = True
        atoms.set_initial_magnetic_moments(len(atoms) * [0.0])
        return atoms

    def _apply_write_read(self, constraint) -> Atoms:
        atoms_ref = self._make_atoms_ref()
        atoms_ref.set_constraint(constraint)

        pseudopotentials = {
            "H": "h_lda_v1.2.uspp.F.UPF",
            "O": "o_lda_v1.2.uspp.F.UPF",
        }
        buf = io.StringIO()
        write_espresso_in(buf, atoms_ref, pseudopotentials=pseudopotentials)
        buf.seek(0)
        atoms = read_espresso_in(buf)

        assert not compare_atoms(atoms_ref, atoms)

        print(atoms_ref.constraints, atoms.constraints)

        return atoms

    def test_fix_atoms(self):
        """Test FixAtoms"""
        constraint = FixAtoms(indices=(1, 2))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixAtoms)
        assert all(atoms.constraints[0].index == constraint.index)

    def test_fix_cartesian_line(self):
        """Test FixCartesian along line"""
        # moved only along the z direction
        constraint = FixCartesian(0, mask=(1, 1, 0))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixCartesian)
        assert all(atoms.constraints[0].index == constraint.index)

    def test_fix_cartesian_plane(self):
        """Test FixCartesian in plane"""
        # moved only in the yz plane
        constraint = FixCartesian((1, 2), mask=(1, 0, 0))
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixCartesian)
        assert all(atoms.constraints[0].index == constraint.index)

    def test_fix_cartesian_multiple(self):
        """Test multiple FixCartesian"""
        constraint = [FixCartesian(1), FixCartesian(2)]
        atoms = self._apply_write_read(constraint)

        assert len(atoms.constraints) == 1
        assert isinstance(atoms.constraints[0], FixAtoms)
        assert atoms.constraints[0].index.tolist() == [1, 2]

    def test_fix_scaled(self):
        """Test FixScaled"""
        constraint = FixScaled(0, mask=(1, 1, 0))
        with pytest.raises(UserWarning):
            self._apply_write_read(constraint)
