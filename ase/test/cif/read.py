import tempfile
import unittest

from ase.io.cif import read_cif
try:
    from pycodcif import parse
except ImportError:
    # Skip test if pycodcif installation is broken:
    raise unittest.SkipTest

cif = """data_Quartz
loop_
_publ_author_name
'Glinnemann J'
'King H E'
'Schulz H'
'Hahn T'
'La Placa S J'
'Dacol F'
_journal_name_full "Zeitschrift fur Kristallographie"
_journal_volume 198
_journal_year 1992
_journal_page_first 177
_journal_page_last 212
_publ_section_title
;Crystal structures of the low-temperature quartz-type phases of SiO2 and GeO2 at elevated pressure

 P = 10.2GPa = 102 kbar
;
_chemical_formula_sum 'Si O2'
_cell_length_a 4.604
_cell_length_b 4.604
_cell_length_c 5.207
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 120
_cell_volume 95.585
_symmetry_space_group_name_H-M 'P 31 2 1'
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Si   0.44580   0.00000   0.00000
O    0.39510   0.30310  -0.09210
"""

with tempfile.NamedTemporaryFile() as temp:
    temp.write(cif.encode("latin-1"))

    temp.seek(0)
    cif_ase = read_cif(temp, 0, reader='ase')

    temp.seek(0)
    cif_pycodcif = read_cif(temp, 0, reader='pycodcif')

    assert [x.__repr__() for x in cif_ase] == [x.__repr__() for x in cif_pycodcif]
