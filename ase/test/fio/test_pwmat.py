import os
import unittest

from ase import Atoms
from ase.io import read

# atom.config content
atom_config_string: str = '''4   atoms
Lattice vector
    3.430000       0.000000       0.000000
    0.000000       3.430000       0.000000
    0.000000       0.000000       3.430000
Position, move_x, move_y, move_z
26	 0.000000000000000	 0.500000000000000	 0.500000000000000  1   1   1
26	 0.500000000000000	 0.000000000000000	 0.500000000000000  1   1   1
26	 0.500000000000000	 0.500000000000000	 0.000000000000000  1   1   1
27	 0.000000000000000	 0.000000000000000	 0.000000000000000  1   1   1
MAGNETIC
26	 3.000000000000000
26	 3.000000000000000
26	 3.000000000000000
27	 2.000000000000000'''

# REPORT CONTENT
report_string: str = '''
 *********************************************
 *********** end of etot.input report ********
 minimum n1,n2,n3 from Ecut2
      29.178       29.178       29.178
 minimum n1L,n2L,n3L from Ecut2L
      29.178       29.178       29.178
 *********************************************


Weighted average num_of_PW for all kpoint=                         1627.080
 ************************************
 The total core charge for core correction      15.65533831      15.65321563
 E_Hxc(eV)         1138.90466139676
 E_ion(eV)        -11284.2078947102
 E_Coul(eV)        3364.86623860396
 E_Hxc+E_ion(eV)  -10145.3032333134
 NONSCF     1          AVE_STATE_ERROR= 0.2764E+01
 NONSCF     2          AVE_STATE_ERROR= 0.9298E+00
 NONSCF     3          AVE_STATE_ERROR= 0.9359E-01
 NONSCF     4          AVE_STATE_ERROR= 0.4637E-02
 NONSCF     5          AVE_STATE_ERROR= 0.5877E-04
 iter=   7   ave_lin=  1.1  iCGmth=   3
 Ef(eV)               = 0.1661158E+02
 err of ug            = 0.6181E-06
 dv_ave, drho_tot     = 0.0000E+00 0.7568E-01
 spin_up;dn;loc_diff  =      36.2992804562       28.7007195438
 E_tot(eV)            = -.14356083220078E+05    -.1436E+05
 -------------------------------------------
 iter=   8   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = 0.1501435E+02
 err of ug            = 0.3437E-04
 dv_ave, drho_tot     = 0.4593E-01 0.4131E-01
 spin_up;dn;loc_diff  =      36.5963461814       28.4036538186
 E_tot(eV)            = -.14352618391141E+05    0.3465E+01
 -------------------------------------------
 iter=   9   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = 0.1497326E+02
 err of ug            = 0.5428E-05
 dv_ave, drho_tot     = 0.1451E-01 0.3521E-01
 spin_up;dn;loc_diff  =      36.6426410275       28.3573589726
 E_tot(eV)            = -.14353014402395E+05    -.3960E+00
 -------------------------------------------
 iter=  10   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = 0.1495162E+02
 err of ug            = 0.2957E-05
 dv_ave, drho_tot     = 0.1056E-01 0.2395E-01
 spin_up;dn;loc_diff  =      36.5429746303       28.4570253697
 E_tot(eV)            = -.14352882662210E+05    0.1317E+00
 -------------------------------------------
 iter=  11   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = 0.1501036E+02
 err of ug            = 0.3360E-05
 dv_ave, drho_tot     = 0.1251E-01 0.4321E-02
 spin_up;dn;loc_diff  =      36.5032717537       28.4967282463
 E_tot(eV)            = -.14352963043135E+05    -.8038E-01
 -------------------------------------------
 iter=  12   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = 0.1500012E+02
 err of ug            = 0.4245E-06
 dv_ave, drho_tot     = 0.1416E-02 0.2897E-02
 spin_up;dn;loc_diff  =      36.5077738762       28.4922261238
 E_tot(eV)            = -.14352965951705E+05    -.2909E-02
 -------------------------------------------
 iter=  13   ave_lin=  4.0  iCGmth=   3
 Ef(eV)               = 0.1499672E+02
 err of ug            = 0.4577E-06
 dv_ave, drho_tot     = 0.1767E-02 0.7071E-03
 spin_up;dn;loc_diff  =      36.5076801668       28.4923198332
 E_tot(eV)            = -.14352968790116E+05    -.2838E-02
 -------------------------------------------
 iter=  14   ave_lin=  2.0  iCGmth=   3
 Ef(eV)               = 0.1499766E+02
 err of ug            = 0.8293E-06
 dv_ave, drho_tot     = 0.2344E-03 0.4764E-03
 spin_up;dn;loc_diff  =      36.5138709976       28.4861290024
 E_tot(eV)            = -.14352968969878E+05    -.1798E-03
 -------------------------------------------
 iter=  15   ave_lin=  2.1  iCGmth=   3
 Ef(eV)               = 0.1499861E+02
 err of ug            = 0.1058E-05
 dv_ave, drho_tot     = 0.4184E-03 0.2651E-03
 spin_up;dn;loc_diff  =      36.5236038227       28.4763961773
 E_tot(eV)            = -.14352969165199E+05    -.1953E-03
 -------------------------------------------
 iter=  16   ave_lin=  2.0  iCGmth=   3
 Ef(eV)               = 0.1499884E+02
 err of ug            = 0.6039E-06
 dv_ave, drho_tot     = 0.1271E-03 0.1675E-03
 spin_up;dn;loc_diff  =      36.5336758016       28.4663241984
 E_tot(eV)            = -.14352969174802E+05    -.9604E-05
 -------------------------------------------
 iter=  17   ave_lin=  2.0  iCGmth=   3
 Ef(eV)               = 0.1500038E+02
 err of ug            = 0.6114E-06
 dv_ave, drho_tot     = 0.1676E-03 0.1129E-03
 spin_up;dn;loc_diff  =      36.5459861911       28.4540138089
 E_tot(eV)            = -.14352969229525E+05    -.5472E-04
 -------------------------------------------
 iter=  18   ave_lin=  1.0  iCGmth=   3
 Ef(eV)               = 0.1499947E+02
 err of ug            = 0.1025E-05
 dv_ave, drho_tot     = 0.8458E-04 0.7175E-04
 spin_up;dn;loc_diff  =      36.5423349081       28.4576650919
 E_tot(eV)            = -.14352969234851E+05    -.5326E-05
 -------------------------------------------
 iter=  19   ave_lin=  1.0  iCGmth=   3
 Ef(eV)               = 0.1500045E+02
 err of ug            = 0.1111E-05
 dv_ave, drho_tot     = 0.9140E-04 0.8759E-05
 spin_up;dn;loc_diff  =      36.5487662432       28.4512337568
 E_tot(eV)            = -.14352969246887E+05    -.1204E-04
 -------------------------------------------
 E_Fermi(eV)=   15.0004524767504
 ---------------------------------------------------
 spin_up;dn;loc_diff  =      36.5487662432       28.4512337568
 ---------------------------------------------------
 Ef(eV)               = 0.1500045E+02
 dvE, dvE(n)-dvE(n-1) = 0.8354E-08 0.1200E-08
 dv_ave, drho_tot     = 0.9140E-04 0.8759E-05
 err of ug            = 0.1111E-05
 ---------------------------------------------------
 ending_scf_reason = tol Rho_err  5.000000000000000E-005
 Ewald        = -.10168860882650E+05
 Alpha        = 0.41410260171389E+03
 E_extV       = 0.00000000000000E+00    0.0000E+00
 E_NSC        = -.11846572320594E+04    0.3324E-01
 E[-rho*V_Hxc]= -.44425894663615E+04    -.3334E-01
 E_Hxc        = 0.10290427376089E+04    0.1265E-01
 -TS          = -.70051384029892E-02    0.8684E-04
 E_tot(eV)    = -.14352969246887E+05    -.1204E-04
 E_tot(Ryd)   = -.10549237908233E+04    -.4423E-06
 ---------------------------------------------------
 ---------------------------------------------------
 occup for:      kpt=1,spin=1,m=(totN/2-2,totN/2+2)
   1.000    0.911    0.001    0.001    0.001
 eigen(eV) for:  kpt=1,spin=1,m=(totN/2-2,totN/2+2)
  14.822   14.953   15.107   15.107   15.107
 ---------------------------------------------------
 ---------------------------------------------------
 E_Hart,E_xc,E_ion =0.32295305154681E+04  -.22004877778593E+04
 E_Hxc+E_ion       =-.99869913990538E+04
 E_kin+E_nonloc    =0.53887874382417E+04
 E_rhoVext,E_IVext     =0.00000000000000E+00  0.00000000000000E+00
 E_psiV,E_dDrho =-.65734408237873E+04  0.00000000000000E+00
 ave(vtot):v0 =-.14682093315657E+02
 ave(V_ion_s(or p,d))=ave(V_Hatree)=0; ave(Vtot)=ave(V_xc)=v0
 ---------------------------------------------------
 *********************************
 Eigen energies are values after setting ave(Vtot)=0
 For Vtot=V_ion+V_Hartree+V_xc, and
 ave(V_ion+V_Hatree)=0, ave(V_xc).ne.0:  E=E+v0
 *********************************
'''


class TestReadPWmat(unittest.TestCase):
    def setUp(self):
        self.atom_config_path: str = "atom.config"
        self.report_path: str = "REPORT"
        with open(self.atom_config_path, 'w', encoding='utf-8') as f:
            f.write(atom_config_string)
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(report_string)

    def test_read_pwmat(self):
        atoms: Atoms = read("atom.config")
        self.assertEqual(len(atoms), 4)

    def test_read_report(self):
        report = read(self.report_path, index=-1)   # Get last step of scf
        self.assertEqual(report.get_potential_energy(), -.14352969246887E+05)

    def tearDown(self):
        if os.path.isfile(self.atom_config_path):
            os.remove(self.atom_config_path)
        if os.path.isfile(self.report_path):
            os.remove(self.report_path)
