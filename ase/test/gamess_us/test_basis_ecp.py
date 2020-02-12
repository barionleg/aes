import numpy as np

from ase.build import molecule
from ase.calculators.gamess_us import GAMESSUS

# NaCl dimer with LANL2DZ basis and ECP.
# Obtained from https://www.basissetexchange.org


basis = dict()

basis['Na'] = """S   2
1         0.4972000             -0.2753574
2         0.0560000              1.0989969
S   1
1         0.0221000              1.0000000
P   2
1         0.6697000             -0.0683845
2         0.0636000              1.0140550
P   1
1         0.0204000              1.0000000"""

basis['Cl'] = """S   2
1         2.2310000             -0.4900589
2         0.4720000              1.2542684
S   1
1         0.1631000              1.0000000
P   2
1         6.2960000             -0.0635641
2         0.6333000              1.0141355
P   1
1         0.1819000              1.0000000"""

ecp = dict()

ecp['Na'] = """NA-ECP GEN    10    2
5     ----- d-ul potential -----
    -10.0000000       1     175.5502590
    -47.4902024       2      35.0516791
    -17.2283007       2       7.9060270
     -6.0637782       2       2.3365719
     -0.7299393       2       0.7799867
5     ----- s-d potential -----
      3.0000000       0     243.3605846
     36.2847626       1      41.5764759
     72.9304880       2      13.2649167
     23.8401151       2       3.6797165
      6.0123861       2       0.9764209
6     ----- p-d potential -----
      5.0000000       0    1257.2650682
    117.4495683       1     189.6248810
    423.3986704       2      54.5247759
    109.3247297       2      13.7449955
     31.3701656       2       3.6813579
      7.1241813       2       0.9461106"""

ecp['Cl'] = """CL-ECP GEN    10    2
5     ----- d-ul potential -----
    -10.0000000       1      94.8130000
     66.2729170       2     165.6440000
    -28.9685950       2      30.8317000
    -12.8663370       2      10.5841000
     -1.7102170       2       3.7704000
5     ----- s-d potential -----
      3.0000000       0     128.8391000
     12.8528510       1     120.3786000
    275.6723980       2      63.5622000
    115.6777120       2      18.0695000
     35.0606090       2       3.8142000
6     ----- p-d potential -----
      5.0000000       0     216.5263000
      7.4794860       1      46.5723000
    613.0320000       2     147.4685000
    280.8006850       2      48.9869000
    107.8788240       2      13.2096000
     15.3439560       2       3.1831000"""


def test_gamess_us_basis_ecp():
    atoms = molecule('NaCl')
    atoms.calc = GAMESSUS(basis=basis, ecp=ecp, label='NaCl')
    e = atoms.get_potential_energy()
    np.testing.assert_allclose(e, -407.32054460869796, atol=1e-3, rtol=1e-3)
