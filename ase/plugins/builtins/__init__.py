"""
This module contains the functionality, that is built-in into ASE
and which is supposed to be used via plugins mechanism.
"""

from ase.register import register_calculator
from ase.io.formats import define_io_format
from ase.visualize.viewers import define_viewer
from ase.register.plugins import get_currently_registered_plugin
import sys
from typing import Callable

"""
note: ase.register.register_io_format is just a wrapper
to define_io_format. We call here define_io_format to
retain the old order of the parameters (since register
begins as all register_.... functions with implementation).
The same holds for the viewers.
"""


def ase_register():
    register_calculators()
    register_formats()
    register_viewers()


def register_calculators():
    register_calculator("ase.calculators.demon.demon.Demon")
    register_calculator("ase.calculators.kim.kim.KIM")
    register_calculator("ase.calculators.openmx.openmx.OpenMX")
    register_calculator("ase.calculators.siesta.siesta.Siesta")
    register_calculator("ase.calculators.turbomole.turbomole.Turbomole")
    register_calculator("ase.calculators.vasp.vasp.Vasp")
    register_calculator("ase.calculators.abinit.Abinit")
    register_calculator("ase.calculators.acemolecule.ACE")
    register_calculator("ase.calculators.acn.ACN")
    register_calculator("ase.calculators.aims.Aims")
    register_calculator("ase.calculators.amber.Amber")
    register_calculator("ase.calculators.castep.Castep")
    register_calculator("ase.calculators.combine_mm.CombineMM")
    register_calculator("ase.calculators.counterions.AtomicCounterIon")
    register_calculator("ase.calculators.cp2k.CP2K")
    register_calculator("ase.calculators.demonnano.DemonNano")
    register_calculator("ase.calculators.dftb.Dftb")
    register_calculator("ase.calculators.dftd3.DFTD3")
    register_calculator("ase.calculators.dmol.DMol3")
    register_calculator("ase.calculators.eam.EAM")
    register_calculator("ase.calculators.elk.ELK")
    register_calculator("ase.calculators.emt.EMT")
    register_calculator("ase.calculators.espresso.Espresso")
    register_calculator("ase.calculators.ff.ForceField")
    register_calculator("ase.calculators.gamess_us.GAMESSUS")
    register_calculator("ase.calculators.gaussian.GaussianDynamics")
    register_calculator("ase.calculators.gromacs.Gromacs")
    register_calculator("ase.calculators.lammpslib.LAMMPSlib")
    register_calculator("ase.calculators.lammpsrun.LAMMPS")
    register_calculator("ase.calculators.lj.LennardJones")
    register_calculator("ase.calculators.mopac.MOPAC")
    register_calculator("ase.calculators.morse.MorsePotential")
    register_calculator("ase.calculators.nwchem.NWChem")
    register_calculator("ase.calculators.octopus.Octopus")
    register_calculator("ase.calculators.onetep.Onetep")
    register_calculator("ase.calculators.orca.ORCA")
    register_calculator("ase.calculators.plumed.Plumed")
    register_calculator("ase.calculators.psi4.Psi4")
    register_calculator("ase.calculators.qchem.QChem")
    register_calculator("ase.calculators.qmmm.SimpleQMMM")
    register_calculator("ase.calculators.qmmm.EIQMMM")
    register_calculator("ase.calculators.qmmm.RescaledCalculator")
    register_calculator("ase.calculators.qmmm.ForceConstantCalculator")
    register_calculator("ase.calculators.qmmm.ForceQMMM")
    register_calculator("ase.calculators.tip3p.TIP3P")
    register_calculator("ase.calculators.tip4p.TIP4P")


def define_to_register(register_function: Callable):

    plugin = get_currently_registered_plugin()

    def register(*args, **kwargs):
        out = register_function(*args, **kwargs)
        out.plugin = plugin
        out.register()

    return register


def register_formats():
    # We define all the IO formats below.  Each IO format has a code,
    # such as '1F', which defines some of the format's properties:
    #
    # 1=single atoms object
    # +=multiple atoms objects
    # F=accepts a file-descriptor
    # S=needs a file-name str
    # B=like F, but opens in binary mode

    F = define_to_register(define_io_format)
    F('abinit-gsr', 'ABINIT GSR file', '1S',
      module='abinit', glob='*o_GSR.nc')
    F('abinit-in', 'ABINIT input file', '1F',
      module='abinit', magic=b'*znucl *')
    F('abinit-out', 'ABINIT output file', '1F',
      module='abinit', magic=b'*.Version * of ABINIT')
    F('aims', 'FHI-aims geometry file', '1S', ext='in')
    F('aims-output', 'FHI-aims output', '+S',
      module='aims', magic=b'*Invoking FHI-aims ...')
    F('bundletrajectory', 'ASE bundle trajectory', '+S')
    F('castep-castep', 'CASTEP output file', '+F',
      module='castep', ext='castep')
    F('castep-cell', 'CASTEP geom file', '1F',
      module='castep', ext='cell')
    F('castep-geom', 'CASTEP trajectory file', '+F',
      module='castep', ext='geom')
    F('castep-md', 'CASTEP molecular dynamics file', '+F',
      module='castep', ext='md')
    F('castep-phonon', 'CASTEP phonon file', '1F',
      module='castep', ext='phonon')
    F('cfg', 'AtomEye configuration', '1F')
    F('cif', 'CIF-file', '+B', ext='cif')
    F('cmdft', 'CMDFT-file', '1F', glob='*I_info')
    F('cjson', 'Chemical json file', '1F', ext='cjson')
    F('cp2k-dcd', 'CP2K DCD file', '+B',
      module='cp2k', ext='dcd')
    F('cp2k-restart', 'CP2K restart file', '1F',
      module='cp2k', ext='restart')
    F('crystal', 'Crystal fort.34 format', '1F',
      ext=['f34', '34'], glob=['f34', '34'])
    F('cube', 'CUBE file', '1F', ext='cube')
    F('dacapo-text', 'Dacapo text output', '1F',
      module='dacapo', magic=b'*&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n')
    F('db', 'ASE SQLite database file', '+S')
    F('dftb', 'DftbPlus input file', '1S', magic=b'Geometry')
    F('dlp4', 'DL_POLY_4 CONFIG file', '1F',
      module='dlp4', ext='config', glob=['*CONFIG*'],
      allowed_pbc=["000", "110", "111"])
    F('dlp-history', 'DL_POLY HISTORY file', '+F',
      module='dlp4', glob='HISTORY')
    F('dmol-arc', 'DMol3 arc file', '+S',
      module='dmol', ext='arc')
    F('dmol-car', 'DMol3 structure file', '1S',
      module='dmol', ext='car')
    F('dmol-incoor', 'DMol3 structure file', '1S',
      module='dmol')
    F('elk', 'ELK atoms definition from GEOMETRY.OUT', '1F',
      glob=['GEOMETRY.OUT'])
    F('elk-in', 'ELK input file', '1F', module='elk')
    F('eon', 'EON CON file', '+F',
      ext='con')
    F('eps', 'Encapsulated Postscript', '1S')
    F('espresso-in', 'Quantum espresso in file', '1F',
      module='espresso', ext='pwi', magic=[b'*\n&system', b'*\n&SYSTEM'])
    F('espresso-out', 'Quantum espresso out file', '+F',
      module='espresso', ext=['pwo', 'out'], magic=b'*Program PWSCF')
    F('exciting', 'exciting input', '1F', module='exciting', glob='input.xml')
    F('exciting', 'exciting output', '1F', module='exciting', glob='INFO.out')
    F('extxyz', 'Extended XYZ file', '+F', ext='xyz')
    F('findsym', 'FINDSYM-format', '+F')
    F('gamess-us-out', 'GAMESS-US output file', '1F',
      module='gamess_us', magic=b'*GAMESS')
    F('gamess-us-in', 'GAMESS-US input file', '1F',
      module='gamess_us')
    F('gamess-us-punch', 'GAMESS-US punchcard file', '1F',
      module='gamess_us', magic=b' $DATA', ext='dat')
    F('gaussian-in', 'Gaussian com (input) file', '1F',
      module='gaussian', ext=['com', 'gjf'])
    F('gaussian-out', 'Gaussian output file', '+F',
      module='gaussian', ext='log', magic=b'*Entering Gaussian System')
    F('acemolecule-out', 'ACE output file', '1S',
      module='acemolecule')
    F('acemolecule-input', 'ACE input file', '1S',
      module='acemolecule')
    F('gen', 'DFTBPlus GEN format', '1F')
    F('gif', 'Graphics interchange format', '+S',
      module='animation')
    F('gpaw-out', 'GPAW text output', '+F',
      magic=b'*  ___ ___ ___ _ _ _')
    F('gpumd', 'GPUMD input file', '1F', glob='xyz.in')
    F('gpw', 'GPAW restart-file', '1S',
      magic=[b'- of UlmGPAW', b'AFFormatGPAW'])
    F('gromacs', 'Gromacs coordinates', '1F',
      ext='gro')
    F('gromos', 'Gromos96 geometry file', '1F', ext='g96')
    F('html', 'X3DOM HTML', '1F', module='x3d')
    F('json', 'ASE JSON database file', '+F', ext='json', module='db')
    F('jsv', 'JSV file format', '1F')
    F('lammps-dump-text', 'LAMMPS text dump file', '+F',
      module='lammpsrun', magic_regex=b'.*?^ITEM: TIMESTEP$')
    F('lammps-dump-binary', 'LAMMPS binary dump file', '+B',
      module='lammpsrun')
    F('lammps-data', 'LAMMPS data file', '1F', module='lammpsdata',
      encoding='ascii')
    F('magres', 'MAGRES ab initio NMR data file', '1F')
    F('mol', 'MDL Molfile', '1F')
    F('mp4', 'MP4 animation', '+S',
      module='animation')
    F('mustem', 'muSTEM xtl file', '1F',
      ext='xtl')
    F('mysql', 'ASE MySQL database file', '+S',
      module='db')
    F('netcdftrajectory', 'AMBER NetCDF trajectory file', '+S',
      magic=b'CDF')
    F('nomad-json', 'JSON from Nomad archive', '+F',
      ext='nomad-json')
    F('nwchem-in', 'NWChem input file', '1F',
      module='nwchem', ext='nwi')
    F('nwchem-out', 'NWChem output file', '+F',
      module='nwchem', ext='nwo',
      magic=b'*Northwest Computational Chemistry Package')
    F('octopus-in', 'Octopus input file', '1F',
      module='octopus', glob='inp')
    F('onetep-out', 'ONETEP output file', '+F',
      module='onetep',
      magic=b'*Linear-Scaling Ab Initio Total Energy Program*')
    F('onetep-in', 'ONETEP input file', '1F',
      module='onetep',
      magic=[b'*lock species ',
             b'*LOCK SPECIES ',
             b'*--- INPUT FILE ---*'])
    F('proteindatabank', 'Protein Data Bank', '+F',
      ext='pdb')
    F('png', 'Portable Network Graphics', '1B')
    F('postgresql', 'ASE PostgreSQL database file', '+S', module='db')
    F('pov', 'Persistance of Vision', '1S')
    # prismatic: Should have ext='xyz' if/when multiple formats can have
    # the same extension
    F('prismatic', 'prismatic and computem XYZ-file', '1F')
    F('py', 'Python file', '+F')
    F('sys', 'qball sys file', '1F')
    F('qbox', 'QBOX output file', '+F',
      magic=b'*:simulation xmlns:')
    F('res', 'SHELX format', '1S', ext='shelx')
    F('rmc6f', 'RMCProfile', '1S', ext='rmc6f')
    F('sdf', 'SDF format', '1F')
    F('siesta-xv', 'Siesta .XV file', '1F',
      glob='*.XV', module='siesta')
    F('struct', 'WIEN2k structure file', '1S', module='wien2k')
    F('struct_out', 'SIESTA STRUCT file', '1F', module='siesta')
    F('traj', 'ASE trajectory', '+B', module='trajectory', ext='traj',
      magic=[b'- of UlmASE-Trajectory', b'AFFormatASE-Trajectory'])
    F('turbomole', 'TURBOMOLE coord file', '1F', glob='coord',
      magic=b'$coord')
    F('turbomole-gradient', 'TURBOMOLE gradient file', '+F',
      module='turbomole', glob='gradient', magic=b'$grad')
    F('v-sim', 'V_Sim ascii file', '1F', ext='ascii')
    F('vasp', 'VASP POSCAR/CONTCAR', '1F',
      ext='poscar', glob=['*POSCAR*', '*CONTCAR*', '*CENTCAR*'])
    F('vasp-out', 'VASP OUTCAR file', '+F',
      module='vasp', glob='*OUTCAR*')
    F('vasp-xdatcar', 'VASP XDATCAR file', '+F',
      module='vasp', glob='*XDATCAR*')
    F('vasp-xml', 'VASP vasprun.xml file', '+F',
      module='vasp', glob='*vasp*.xml')
    F('vti', 'VTK XML Image Data', '1F', module='vtkxml')
    F('vtu', 'VTK XML Unstructured Grid', '1F', module='vtkxml', ext='vtu')
    F('wout', 'Wannier90 output', '1F', module='wannier90')
    F('x3d', 'X3D', '1S')
    F('xsd', 'Materials Studio file', '1F')
    F('xsf', 'XCrySDen Structure File', '+F',
      magic=[b'*\nANIMSTEPS', b'*\nCRYSTAL', b'*\nSLAB', b'*\nPOLYMER',
             b'*\nMOLECULE', b'*\nATOMS'])
    F('xtd', 'Materials Studio file', '+F')
    # xyz: No `ext='xyz'` in the definition below.
    #      The .xyz files are handled by the extxyz module by default.
    F('xyz', 'XYZ-file', '+F')


def register_viewers():
    F = define_to_register(define_viewer)

    F("ase", "View atoms using ase gui.")
    F("ngl", "View atoms using nglview.")
    F("mlab", "View atoms using matplotlib.")
    F("sage", "View atoms using sage.")
    F("x3d", "View atoms using x3d.")

    # CLI viweers that are internally supported
    F(
        "avogadro", "View atoms using avogradro.", cli=True, fmt="cube",
        argv=["avogadro"]
    )
    F(
        "ase_gui_cli", "View atoms using ase gui.", cli=True, fmt="traj",
        argv=[sys.executable, '-m', 'ase.gui'],
    )
    F(
        "gopenmol",
        "View atoms using gopenmol.",
        cli=True,
        fmt="extxyz",
        argv=["runGOpenMol"],
    )
    F(
        "rasmol",
        "View atoms using rasmol.",
        cli=True,
        fmt="proteindatabank",
        argv=["rasmol", "-pdb"],
    )
    F("vmd", "View atoms using vmd.", cli=True, fmt="cube", argv=["vmd"])
    F(
        "xmakemol",
        "View atoms using xmakemol.",
        cli=True,
        fmt="extxyz",
        argv=["xmakemol", "-f"],
    )
