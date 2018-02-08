from __future__ import print_function
"""Module for calculating phonons of periodic systems."""

import sys
import pickle
from math import pi, sqrt
from os import remove
from os.path import isfile

import numpy as np
import numpy.linalg as la
import numpy.fft as fft

import ase.units as units
from ase.parallel import rank, world
from ase.dft import monkhorst_pack
from ase.io.trajectory import Trajectory
from ase.utils import opencew, pickleload, basestring
from ase.spacegroup import Spacegroup, get_spacegroup


class Displacement:
    """Abstract base class for phonon and el-ph supercell calculations.

    Both phonons and the electron-phonon interaction in periodic systems can be
    calculated with the so-called finite-displacement method where the
    derivatives of the total energy and effective potential are obtained from
    finite-difference approximations, i.e. by displacing the atoms. This class
    provides the required functionality for carrying out the calculations for
    the different displacements in its ``run`` member function.

    Derived classes must overwrite the ``__call__`` member function which is
    called for each atomic displacement.

    """

    def __init__(self, atoms, calc=None, supercell=(1, 1, 1), name=None,
                 delta=0.01, refcell=None):
        """Init with an instance of class ``Atoms`` and a calculator.

        Parameters:

        atoms: Atoms object
            The atoms to work on.
        calc: Calculator
            Calculator for the supercell calculation.
        supercell: tuple
            Size of supercell given by the number of repetitions (l, m, n) of
            the small unit cell in each direction.
        name: str
            Base name to use for files.
        delta: float
            Magnitude of displacement in Ang.
        refcell: str
            Reference cell in which the atoms will be displaced. If ``None``,
            corner cell in supercell is used. If ``str``, cell in the center of
            the supercell is used.

        """

        # Store atoms and calculator
        self.atoms = atoms
        self.calc = calc

        # Displace all atoms in the unit cell by default
        self.indices = np.arange(len(atoms))
        self.name = name
        self.delta = delta
        self.N_c = supercell

        # Reference cell offset
        if refcell is None:
            # Corner cell
            self.offset = 0
        else:
            # Center cell
            N_c = self.N_c
            self.offset = (N_c[0] // 2 * (N_c[1] * N_c[2]) +
                           N_c[1] // 2 * N_c[2] +
                           N_c[2] // 2)

    def __call__(self, *args, **kwargs):
        """Member function called in the ``run`` function."""

        raise NotImplementedError("Implement in derived classes!.")

    def set_atoms(self, atoms):
        """Set the atoms to vibrate.

        Parameters:

        atoms: list
            Can be either a list of strings, ints or ...

        """

        assert isinstance(atoms, list)
        assert len(atoms) <= len(self.atoms)

        if isinstance(atoms[0], basestring):
            assert np.all([isinstance(atom, basestring) for atom in atoms])
            sym_a = self.atoms.get_chemical_symbols()
            # List for atomic indices
            indices = []
            for type in atoms:
                indices.extend([a for a, atom in enumerate(sym_a)
                                if atom == type])
        else:
            assert np.all([isinstance(atom, int) for atom in atoms])
            indices = atoms

        self.indices = indices

    def lattice_vectors(self):
        """Return lattice vectors for cells in the supercell."""

        # Lattice vectors relevative to the reference cell
        R_cN = np.indices(self.N_c).reshape(3, -1)
        N_c = np.array(self.N_c)[:, np.newaxis]
        if self.offset == 0:
            R_cN += N_c // 2
            R_cN %= N_c
        R_cN -= N_c // 2

        return R_cN
        
    def run_eq(self, supercell):
        """Run the calculation for the equilibrium lattice
        
        input:
        
          self:      ASE Phonon/Displacement object with Atoms and Calculator
          supercell: an ASE Atoms object, supercell
        
        return:
          output: forces (ndarray)
        """
        
        # Do calculation on equilibrium structure
        filename = self.name + '.eq.pckl'
    
        fd = opencew(filename)
        if fd is not None:
            # Call derived class implementation of __call__
            output = self.__call__(supercell)
                        
            # Write output to file
            if rank == 0:
                pickle.dump(output, fd, protocol=2)
                sys.stdout.write('Writing %s\n' % filename)
                fd.close()
                # check forces
                try:
                    fmax = output.max()
                    fmin = output.min()
                    sys.stdout.write('[ASE] Equilibrium forces min=%g max=%g\n' % (fmin, fmax))
                except AttributeError:
                    sys.stdout.write('[ASE] output has no min/max (list)\n');
                    pass
            sys.stdout.flush()
        else:
            # read previous data
            output = pickle.load(open(filename))
            
        return output

    def run(self, single=True, difference='central'):
        """Run the calculations for the required displacements.
        This function uses the Spacegroup in order to speed-up the calculation.
        Every iteration is checked against symmetry operators. In case other moves 
        are imaged from a symmetry, the pickle files are written in advance, which 
        then allows to skip subsequent steps.
    
        This will do a calculation for 6 displacements per atom, +-x, +-y, and
        +-z. Only those calculations that are not already done will be
        started. Be aware that an interrupted calculation may produce an empty
        file (ending with .pckl), which must be deleted before restarting the
        job. Otherwise the calculation for that displacement will not be done.
        
        This implementation is the same as legacy ASE, but allows to select the type of
        force gradient to use (difference). Also, it does make use of the spacegroup 
        to lower the number of displacements, using the strategy adopted in PHON 
        (D. Alfe). It is not as good as PhonoPy, but remains much simpler.
        
        input:
        
          self:   ASE Phonon/Displacement object with Atoms and Calculator
          single: when True, the forces are computed only for a single step, and then
                  exit. This allows to split the loop in independent iterations. When
                  calling again the 'run' method, already computed steps are ignored,
                  missing steps are completed, until no more are needed. When set to
                  False, all steps are done in a row.
          difference: the method to use for the force difference (gradient) computation.
                  can be 'central','forward','backward'. The central difference is 
                  more accurate but requires twice as many force calculations.
        
        output:
        
          True when a calculation step was performed, False otherwise or no more is needed.
          nb_of_iterations: the number of steps remaining
    
        """
        
        # prepare to use symmetry operations
        # check if a spacegroup is defined, else find it
        if 'spacegroup' not in self.atoms.info or self.atoms.info["spacegroup"] is None:
            sg    = get_spacegroup(self.atoms)
    
        if difference == 'backward':
            signs = [-1]    # only compute one side, backward difference
        elif difference == 'forward':
            signs = [1]     # only compute one side, forward difference
        else:
            signs = [-1,1]   # use central difference
    
        # Atoms in the supercell -- repeated in the lattice vector directions
        # beginning with the last
        supercell = self.atoms * self.N_c
        
        # Set calculator if provided
        assert self.calc is not None, "Provide calculator in Phonon __init__ method"
        supercell.set_calculator(self.calc)
        
        # when not central difference, we check if the equilibrium forces are small
        # and will use the '0' forces in gradient
        if len(signs) == 1 and not _isfile_parallel(self.name + '.eq.pckl'): 
            # Do calculation on equilibrium structure
            run_eq(self, supercell)
            if single:
                return len(signs)*len(self.indices)*3 # and some more iterations may be required
    
        # Positions of atoms to be displaced in the reference cell
        natoms = len(self.atoms)
        offset = natoms * self.offset
        pos    = supercell.positions[offset: offset + natoms].copy()
        pos0   = supercell.positions.copy()
        
        L    = supercell.get_cell()     # lattice      cell       = at
        invL = np.linalg.inv(L)         # no 2*pi multiplier here = inv(at) = bg.T
        
        # number of iterations left
        nb_of_iterations = len(signs)*len(self.indices)*3
        
        for sign in signs:  # +-
            # guess displacements and rotations
            dxlist, rotlist = _get_displacements(self, sign)
            
            for a in self.indices:    # atom index to move in cell
            
                # check if this step has been done before (3 files written per step)
                filename = '%s.%d%s%s.pckl' % \
                           (self.name, a, 'xyz'[0], ' +-'[sign])
                if _isfile_parallel(filename):
                        # Skip if this move ('x') is already done.
                        nb_of_iterations -= 3
                        continue
                        
                # we determine the Wigner-Seitz cell around atom 'a' (Cartesian)
                ws = _get_wigner_seitz(supercell, a, 1)
                
                # compute the forces for 3 independent moves
                force0 = None
                force1 = []
                for index in range(len(dxlist)):
                    # we first determine a set of independent displacements, using 
                    # as much as possible the symmetry operators.
                    disp = dxlist[index]
                    rot  = rotlist[index]
                    
                    if force0 is not None and rot is not None:  # re-use previous forces
                        # we will now apply the rotation to the force array
                        output = _run_force1_rot(force0, ws, invL, rot, symprec=1e-6)
                        if output is None:
                            # failed using symmetry to derive force. Trigger full computation.
                            force0 = None
                        elif rank == 0:
                            print("[ASE] Imaging atom #%-3i %-3s    to " % \
                                (offset + a, supercell.get_chemical_symbols()[a]), pos[a] + disp, \
                                " (Angs) using rotation:")
                            print(rot)
                            
                    if force0 is None or rot is None: # compute forces
                        # move atom 'a' by 'disp'
                        supercell.positions[offset + a] = pos[a] + disp
                        if rank == 0:
                            print("[ASE] Moving  atom #%-3i %-3s    to " % \
                                (offset + a, supercell.get_chemical_symbols()[a]), pos[a] + disp, " (Angs)")
                            
                        # Call derived class implementation of __call__
                        output = self.__call__(supercell)
                        
                        # Return to initial positions
                        supercell.positions[offset + a] = pos[a]
                        
                        # store the forces for subsequent moves using symmetry
                        force0 = output
                    
                    # append the forces to the force1 list
                    if output is None:
                        print("[ASE] Warning: force1 is None !!")
    
                    force1.append(output)
                    
                # when exiting the for 'i' loop, we have 3 independent 'force1' array
                # derive a Cartesian basis, and write pickle files
                force2 = _run_force2(self, dxlist, force1, a, sign, symprec=1e-6)
                
                nb_of_iterations -= 3
                
                # then we derive the Cartesian basis 'force2' array and write files
                if single:
                    return nb_of_iterations # and some more iterations may be required
    
        return 0  # nothing left to do
        
    def _get_displacements(self, sign):
        """Determine a set of displacements that best make use of crystal symmetries.
        
           input:
               self:   ASE phonon/Displacement containing an Atoms and a Spacegroup.
               sign:   -1 or +1, to indicate in which direction we move the atoms
           output:
               dxlist:  list of independent displacements
               rotlist: list of corresponding rotations from 'disp'
        """
        
        dxlist = [] # hold list of lists for tentative displacements and rotations
        rotlist= []
        L      = self.atoms.get_cell()
        sg     = self.atoms.info["spacegroup"]    # spacegroup for e.g. spglib
        
        # Determine displacement vectors to use (list[3]). 
        # We look for the equivalent displacements. We only add when symmetry 
        # operators provide equivalent sites (saves time).
        
        # we try with axes [xyz] or lattice axes that have most symmetries
        for i in range(6):
            if i < 3: # [xyz] directions
                disp     = np.asarray([0.0,0,0])
                disp[i] += sign * self.delta # in Angs, Cartesian
            else:     # lattice cell axes
                disp = L[i-3]/np.linalg.norm(L[i-3]) * sign * self.delta
            # test if symmetries can generate other independent moves
            this_dx, this_rot = _rotated_displacements(disp, sg)
            dxlist.append(this_dx)  # append lists
            rotlist.append(this_rot)
        
        # now we sort the lists by length of the sublists, i.e. select the moves 
        # that generate most equivalent moves by symmetry
                 
        # get the index to use for sorting (decreasing order)
        lenlist = [len(x) for x in dxlist]
        order   = sorted(range(len(lenlist)), key=lambda k: lenlist[k], reverse=True)
        # reorder lists by decreasing size
        dxlist  = [ dxlist[j]  for j in order]
        rotlist = [ rotlist[j] for j in order]
        # now catenate all lists
        dxlist  = [j for i in dxlist  for j in i]
        rotlist = [j for i in rotlist for j in i]
        
        # and finally extract iteratively 3 vectors which are independent
        dxlist2 = []
        rotlist2= []
        for index in range(len(dxlist)):
            if _move_is_independent(dxlist2, dxlist[index]):
                dxlist2.append(dxlist[index])
                rotlist2.append(rotlist[index])
            if len(dxlist2) == 3:
                break
        
        # return only the first 3 independent entries
        return dxlist2, rotlist2
        
    def _run_force2(self, dxlist, force1, a, sign, symprec=1e-6):  
        """From a given force set, we derive the forces in Cartesian coordinates
           
           code outrageously adapted from PHON/src/set_forces by D. Alfe
           
           input:
              self:   ASE Phonon/Displacement object with Atoms and Calculator
              dxlist: displacement list, in Cartesian coordinates
              force1: the forces list (3 moves) determined for dxlist
              a:      the index of the atom being moved
              sign:   the [+-] directions to move ([0,1] -> [+,-])
              symprec:the precision for comparing positions
              
           output:
              The forces for Cartesian displacements along [xyz]
        
        """
    
        # get the 3 displacement vectors (rows) and normalise
        dxnorm   = np.asarray([dx/np.linalg.norm(dx) for dx in dxlist])
        invdx    = np.linalg.inv(dxnorm)
        identity = np.diag([1,1,1])
            
        if np.linalg.norm(invdx + identity) < symprec: # inv(dx) = -I -> I
            invdx=identity
        # project the 3 forces into that Cartesian basis
        force2 = []
        for i in range(len(dxlist)):
            force2.append(invdx[i,0] * force1[0] \
                        + invdx[i,1] * force1[1] \
                        + invdx[i,2] * force1[2])
        # write the pickle files assuming we have now [xyz]
        for i in range(3):      # xyz
            # skip if the pickle already exists
            filename = '%s.%d%s%s.pckl' % \
                   (self.name, a, 'xyz'[i], ' +-'[sign])
            if _isfile_parallel(filename):
                # Skip if already done. Also the case for initial 'disp/dx'
                continue
            # write the pickle for the current Cartesian axis
            fd = opencew(filename)
            if rank == 0:
                pickle.dump(force2[i], fd, protocol=2)
                sys.stdout.write('Writing %s\n' % filename)
                fd.close()
                sys.stdout.flush()
        
        return force2

    def clean(self):
        """Delete generated pickle files."""

        if isfile(self.name + '.eq.pckl'):
            remove(self.name + '.eq.pckl')

        for a in self.indices:
            for i in 'xyz':
                for sign in '-+':
                    name = '%s.%d%s%s.pckl' % (self.name, a, i, sign)
                    if isfile(name):
                        remove(name)


class Phonons(Displacement):
    """Class for calculating phonon modes using the finite displacement method.

    The matrix of force constants is calculated from the finite difference
    approximation to the first-order derivative of the atomic forces as::

                            2             nbj   nbj
                nbj        d E           F-  - F+
               C     = ------------ ~  -------------  ,
                mai     dR   dR          2 * delta
                          mai  nbj

    where F+/F- denotes the force in direction j on atom nb when atom ma is
    displaced in direction +i/-i. The force constants are related by various
    symmetry relations. From the definition of the force constants it must
    be symmetric in the three indices mai::

                nbj    mai         bj        ai
               C    = C      ->   C  (R ) = C  (-R )  .
                mai    nbj         ai  n     bj   n

    As the force constants can only depend on the difference between the m and
    n indices, this symmetry is more conveniently expressed as shown on the
    right hand-side.

    The acoustic sum-rule::

                           _ _
                aj         \    bj
               C  (R ) = -  )  C  (R )
                ai  0      /__  ai  m
                          (m, b)
                            !=
                          (0, a)

    Ordering of the unit cells illustrated here for a 1-dimensional system (in
    case ``refcell=None`` in constructor!):

    ::

               m = 0        m = 1        m = -2        m = -1
           -----------------------------------------------------
           |            |            |            |            |
           |        * b |        *   |        *   |        *   |
           |            |            |            |            |
           |   * a      |   *        |   *        |   *        |
           |            |            |            |            |
           -----------------------------------------------------

    Example:

    >>> from ase.build import bulk
    >>> from ase.phonons import Phonons
    >>> from gpaw import GPAW, FermiDirac
    >>> atoms = bulk('Si', 'diamond', a=5.4)
    >>> calc = GPAW(kpts=(5, 5, 5),
                    h=0.2,
                    occupations=FermiDirac(0.))
    >>> ph = Phonons(atoms, calc, supercell=(5, 5, 5))
    >>> ph.run()
    >>> ph.read(method='frederiksen', acoustic=True)

    """

    def __init__(self, *args, **kwargs):
        """Initialize with base class args and kwargs."""

        if 'name' not in kwargs.keys():
            kwargs['name'] = "phonon"

        Displacement.__init__(self, *args, **kwargs)

        # Attributes for force constants and dynamical matrix in real space
        self.C_N = None  # in units of eV / Ang**2
        self.D_N = None  # in units of eV / Ang**2 / amu

        # Attributes for born charges and static dielectric tensor
        self.Z_avv = None
        self.eps_vv = None

    def __call__(self, atoms_N):
        """Calculate forces on atoms in supercell."""

        # Calculate forces
        forces = atoms_N.get_forces()

        return forces

    def check_eq_forces(self):
        """Check maximum size of forces in the equilibrium structure."""

        fname = '%s.eq.pckl' % self.name
        feq_av = pickleload(open(fname, 'rb'))

        fmin = feq_av.max()
        fmax = feq_av.min()
        i_min = np.where(feq_av == fmin)
        i_max = np.where(feq_av == fmax)

        return fmin, fmax, i_min, i_max

    def read_born_charges(self, name=None, neutrality=True):
        """Read Born charges and dieletric tensor from pickle file.

        The charge neutrality sum-rule::

                   _ _
                   \    a
                    )  Z   = 0
                   /__  ij
                    a

        Parameters:

        neutrality: bool
            Restore charge neutrality condition on calculated Born effective
            charges.

        """

        # Load file with Born charges and dielectric tensor for atoms in the
        # unit cell
        if name is None:
            filename = '%s.born.pckl' % self.name
        else:
            filename = name

        with open(filename, 'rb') as fd:
            Z_avv, eps_vv = pickleload(fd)

        # Neutrality sum-rule
        if neutrality:
            Z_mean = Z_avv.sum(0) / len(Z_avv)
            Z_avv -= Z_mean

        self.Z_avv = Z_avv[self.indices]
        self.eps_vv = eps_vv

    def read(self, method='Frederiksen', symmetrize=3, acoustic=True,
             cutoff=None, born=False, **kwargs):
        """Read forces from pickle files and calculate force constants.

        Extra keyword arguments will be passed to ``read_born_charges``.
        
        This implementation is similar to the ASE legacy, but can make use of different
        gradient estimates, depending on what is available on disk (pickles).
        Can use:
          displacement .[xyz]+
          displacement .[xyz]-
          equilibrium  .qe

        Parameters:

        method: str
            Specify method for evaluating the atomic forces.
        symmetrize: int
            Symmetrize force constants (see doc string at top) when
            ``symmetrize != 0`` (default: 3). Since restoring the acoustic sum
            rule breaks the symmetry, the symmetrization must be repeated a few
            times until the changes a insignificant. The integer gives the
            number of iterations that will be carried out.
        acoustic: bool
            Restore the acoustic sum rule on the force constants.
        cutoff: None or float
            Zero elements in the dynamical matrix between atoms with an
            interatomic distance larger than the cutoff.
        born: bool
            Read in Born effective charge tensor and high-frequency static
            dielelctric tensor from file.

        """

        # proceed with pure ASE 'Phonon' object.
        method = method.lower()
        assert method in ['standard', 'frederiksen']
        if cutoff is not None:
            cutoff = float(cutoff)
            
        # Read Born effective charges and optical dielectric tensor
        if born:
            phonon.read_born_charges(**kwargs)
        
        # Number of atoms
        natoms = len(phonon.indices)
        # Number of unit cells
        N = numpy.prod(phonon.N_c)
        # Matrix of force constants as a function of unit cell index in units
        # of eV / Ang**2
        C_xNav = numpy.empty((natoms * 3, N, natoms, 3), dtype=float)
        
        # get equilibrium forces (if any)
        filename = phonon.name + '.eq.pckl'
        feq = 0
        if isfile(filename):
            feq = pickle.load(open(filename))
            if method == 'frederiksen':
                for i, a in enumerate(phonon.indices):
                    feq[a] -= feq.sum(0)

        # Loop over all atomic displacements and calculate force constants
        for i, a in enumerate(phonon.indices):
            for j, v in enumerate('xyz'):
                # Atomic forces for a displacement of atom a in direction v
                basename = '%s.%d%s' % (phonon.name, a, v)
                
                if isfile(basename + '-.pckl'):
                    fminus_av = pickle.load(open(basename + '-.pckl'))
                else:
                    fminus_av = None
                if isfile(basename + '+.pckl'):
                    fplus_av = pickle.load(open(basename + '+.pckl'))
                else:
                    fplus_av = None
                
                if method == 'frederiksen': # translational invariance
                    if fminus_av is not None:
                        fminus_av[a] -= fminus_av.sum(0)
                    if fplus_av is not None:
                        fplus_av[a]  -= fplus_av.sum(0)
                
                if fminus_av is not None and fplus_av is not None:
                    # Finite central difference derivative
                    C_av = (fminus_av - fplus_av)/2
                elif fminus_av is not None:
                    # only the - side is available: forward difference
                    C_av =  fminus_av - feq
                elif fplus_av is not None:
                    # only the + side is available: backward difference
                    C_av = -(fplus_av - feq)

                C_av /= phonon.delta  # gradient

                # Slice out included atoms
                C_Nav = C_av.reshape((N, len(phonon.atoms), 3))[:, phonon.indices]
                index = 3*i + j
                C_xNav[index] = C_Nav

        # Make unitcell index the first and reshape
        C_N = C_xNav.swapaxes(0 ,1).reshape((N,) + (3 * natoms, 3 * natoms))

        # Cut off before symmetry and acoustic sum rule are imposed
        if cutoff is not None:
            phonon.apply_cutoff(C_N, cutoff)
            
        # Symmetrize force constants
        if symmetrize:
            for i in range(symmetrize):
                # Symmetrize
                C_N = phonon.symmetrize(C_N)
                # Restore acoustic sum-rule
                if acoustic:
                    phonon.acoustic(C_N)
                else:
                    break
             
        # Store force constants and dynamical matrix
        phonon.C_N = C_N
        phonon.D_N = C_N.copy()
        
        # Add mass prefactor
        m_a = phonon.atoms.get_masses()
        phonon.m_inv_x = numpy.repeat(m_a[phonon.indices]**-0.5, 3)
        M_inv = numpy.outer(phonon.m_inv_x, phonon.m_inv_x)
        for D in phonon.D_N:
            D *= M_inv

    def symmetrize(self, C_N):
        """Symmetrize force constant matrix."""

        # Number of atoms
        natoms = len(self.indices)
        # Number of unit cells
        N = np.prod(self.N_c)

        # Reshape force constants to (l, m, n) cell indices
        C_lmn = C_N.reshape(self.N_c + (3 * natoms, 3 * natoms))

        # Shift reference cell to center index
        if self.offset == 0:
            C_lmn = fft.fftshift(C_lmn, axes=(0, 1, 2)).copy()
        # Make force constants symmetric in indices -- in case of an even
        # number of unit cells don't include the first
        i, j, k = np.asarray(self.N_c) % 2 - 1
        C_lmn[i:, j:, k:] *= 0.5
        C_lmn[i:, j:, k:] += \
            C_lmn[i:, j:, k:][::-1, ::-1, ::-1].transpose(0, 1, 2, 4, 3).copy()
        if self.offset == 0:
            C_lmn = fft.ifftshift(C_lmn, axes=(0, 1, 2)).copy()

        # Change to single unit cell index shape
        C_N = C_lmn.reshape((N, 3 * natoms, 3 * natoms))

        return C_N

    def acoustic(self, C_N):
        """Restore acoustic sumrule on force constants."""

        # Number of atoms
        natoms = len(self.indices)
        # Copy force constants
        C_N_temp = C_N.copy()

        # Correct atomic diagonals of R_m = (0, 0, 0) matrix
        for C in C_N_temp:
            for a in range(natoms):
                for a_ in range(natoms):
                    C_N[self.offset,
                        3 * a: 3 * a + 3,
                        3 * a: 3 * a + 3] -= C[3 * a: 3 * a + 3,
                                               3 * a_: 3 * a_ + 3]

    def apply_cutoff(self, D_N, r_c):
        """Zero elements for interatomic distances larger than the cutoff.

        Parameters:

        D_N: ndarray
            Dynamical/force constant matrix.
        r_c: float
            Cutoff in Angstrom.

        """

        # Number of atoms and primitive cells
        natoms = len(self.indices)
        N = np.prod(self.N_c)
        # Lattice vectors
        R_cN = self.lattice_vectors()
        # Reshape matrix to individual atomic and cartesian dimensions
        D_Navav = D_N.reshape((N, natoms, 3, natoms, 3))

        # Cell vectors
        cell_vc = self.atoms.cell.transpose()
        # Atomic positions in reference cell
        pos_av = self.atoms.get_positions()

        # Zero elements with a distance to atoms in the reference cell
        # larger than the cutoff
        for n in range(N):
            # Lattice vector to cell
            R_v = np.dot(cell_vc, R_cN[:, n])
            # Atomic positions in cell
            posn_av = pos_av + R_v
            # Loop over atoms and zero elements
            for i, a in enumerate(self.indices):
                dist_a = np.sqrt(np.sum((pos_av[a] - posn_av)**2, axis=-1))
                # Atoms where the distance is larger than the cufoff
                i_a = dist_a > r_c  # np.where(dist_a > r_c)
                # Zero elements
                D_Navav[n, i, :, i_a, :] = 0.0

    def get_force_constant(self):
        """Return matrix of force constants."""

        assert self.C_N is not None

        return self.C_N

    def band_structure(self, path_kc, modes=False, born=False, verbose=True):
        """Calculate phonon dispersion along a path in the Brillouin zone.

        The dynamical matrix at arbitrary q-vectors is obtained by Fourier
        transforming the real-space force constants. In case of negative
        eigenvalues (squared frequency), the corresponding negative frequency
        is returned.

        Frequencies and modes are in units of eV and Ang/sqrt(amu),
        respectively.

        Parameters:

        path_kc: ndarray
            List of k-point coordinates (in units of the reciprocal lattice
            vectors) specifying the path in the Brillouin zone for which the
            dynamical matrix will be calculated.
        modes: bool
            Returns both frequencies and modes when True.
        born: bool
            Include non-analytic part given by the Born effective charges and
            the static part of the high-frequency dielectric tensor. This
            contribution to the force constant accounts for the splitting
            between the LO and TO branches for q -> 0.
        verbose: bool
            Print warnings when imaginary frequncies are detected.

        """

        assert self.D_N is not None
        if born:
            assert self.Z_avv is not None
            assert self.eps_vv is not None

        # Lattice vectors -- ordered as illustrated in class docstring
        R_cN = self.lattice_vectors()

        # Dynamical matrix in real-space
        D_N = self.D_N

        # Lists for frequencies and modes along path
        omega_kl = []
        u_kl = []

        # Reciprocal basis vectors for use in non-analytic contribution
        reci_vc = 2 * pi * la.inv(self.atoms.cell)
        # Unit cell volume in Bohr^3
        vol = abs(la.det(self.atoms.cell)) / units.Bohr**3

        for q_c in path_kc:

            # Add non-analytic part
            if born:
                # q-vector in cartesian coordinates
                q_v = np.dot(reci_vc, q_c)
                # Non-analytic contribution to force constants in atomic units
                qdotZ_av = np.dot(q_v, self.Z_avv).ravel()
                C_na = (4 * pi * np.outer(qdotZ_av, qdotZ_av) /
                        np.dot(q_v, np.dot(self.eps_vv, q_v)) / vol)
                self.C_na = C_na / units.Bohr**2 * units.Hartree
                # Add mass prefactor and convert to eV / (Ang^2 * amu)
                M_inv = np.outer(self.m_inv_x, self.m_inv_x)
                D_na = C_na * M_inv / units.Bohr**2 * units.Hartree
                self.D_na = D_na
                D_N = self.D_N + D_na / np.prod(self.N_c)

            # if np.prod(self.N_c) == 1:
            #
            #     q_av = np.tile(q_v, len(self.indices))
            #     q_xx = np.vstack([q_av]*len(self.indices)*3)
            #     D_m += q_xx

            # Evaluate fourier sum
            phase_N = np.exp(-2.j * pi * np.dot(q_c, R_cN))
            D_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * D_N, axis=0)

            if modes:
                omega2_l, u_xl = la.eigh(D_q, UPLO='U')
                # Sort eigenmodes according to eigenvalues (see below) and
                # multiply with mass prefactor
                u_lx = (self.m_inv_x[:, np.newaxis] *
                        u_xl[:, omega2_l.argsort()]).T.copy()
                u_kl.append(u_lx.reshape((-1, len(self.indices), 3)))
            else:
                omega2_l = la.eigvalsh(D_q, UPLO='U')

            # Sort eigenvalues in increasing order
            omega2_l.sort()
            # Use dtype=complex to handle negative eigenvalues
            omega_l = np.sqrt(omega2_l.astype(complex))

            # Take care of imaginary frequencies
            if not np.all(omega2_l >= 0.):
                indices = np.where(omega2_l < 0)[0]

                if verbose:
                    print('WARNING, %i imaginary frequencies at '
                          'q = (% 5.2f, % 5.2f, % 5.2f) ; (omega_q =% 5.3e*i)'
                          % (len(indices), q_c[0], q_c[1], q_c[2],
                             omega_l[indices][0].imag))

                omega_l[indices] = -1 * np.sqrt(np.abs(omega2_l[indices].real))

            omega_kl.append(omega_l.real)

        # Conversion factor: sqrt(eV / Ang^2 / amu) -> eV
        s = units._hbar * 1e10 / sqrt(units._e * units._amu)
        omega_kl = s * np.asarray(omega_kl)

        if modes:
            return omega_kl, np.asarray(u_kl)

        return omega_kl

    def dos(self, kpts=(10, 10, 10), npts=1000, delta=1e-3,
            indices=None, verbose=True):
        """Calculate phonon dos as a function of energy.

        Parameters:

        qpts: tuple
            Shape of Monkhorst-Pack grid for sampling the Brillouin zone.
        npts: int
            Number of energy points.
        delta: float
            Broadening of Lorentzian line-shape in eV.
        indices: list
            If indices is not None, the atomic-partial dos for the specified
            atoms will be calculated.
        verbose: bool
            Print warnings when imaginary frequncies are detected.

        """

        # Monkhorst-Pack grid
        kpts_kc = monkhorst_pack(kpts)
        N = np.prod(kpts)
        # Get frequencies
        omega_kl = self.band_structure(kpts_kc, verbose=verbose)
        # Energy axis and dos
        omega_e = np.linspace(0., np.amax(omega_kl) + 5e-3, num=npts)
        dos_e = np.zeros_like(omega_e)

        # Sum up contribution from all q-points and branches
        for omega_l in omega_kl:
            diff_el = (omega_e[:, np.newaxis] - omega_l[np.newaxis, :])**2
            dos_el = 1. / (diff_el + (0.5 * delta)**2)
            dos_e += dos_el.sum(axis=1)

        dos_e *= 1. / (N * pi) * 0.5 * delta

        return omega_e, dos_e

    def write_modes(self, q_c, branches=0, kT=units.kB * 300, born=False,
                    repeat=(1, 1, 1), nimages=30, center=False):
        """Write modes to trajectory file.

        Parameters:

        q_c: ndarray
            q-vector of the modes.
        branches: int or list
            Branch index of modes.
        kT: float
            Temperature in units of eV. Determines the amplitude of the atomic
            displacements in the modes.
        born: bool
            Include non-analytic contribution to the force constants at q -> 0.
        repeat: tuple
            Repeat atoms (l, m, n) times in the directions of the lattice
            vectors. Displacements of atoms in repeated cells carry a Bloch
            phase factor given by the q-vector and the cell lattice vector R_m.
        nimages: int
            Number of images in an oscillation.
        center: bool
            Center atoms in unit cell if True (default: False).

        """

        if isinstance(branches, int):
            branch_l = [branches]
        else:
            branch_l = list(branches)

        # Calculate modes
        omega_l, u_l = self.band_structure([q_c], modes=True, born=born)
        # Repeat atoms
        atoms = self.atoms * repeat
        # Center
        if center:
            atoms.center()

        # Here ``Na`` refers to a composite unit cell/atom dimension
        pos_Nav = atoms.get_positions()
        # Total number of unit cells
        N = np.prod(repeat)

        # Corresponding lattice vectors R_m
        R_cN = np.indices(repeat).reshape(3, -1)
        # Bloch phase
        phase_N = np.exp(2.j * pi * np.dot(q_c, R_cN))
        phase_Na = phase_N.repeat(len(self.atoms))

        for l in branch_l:

            omega = omega_l[0, l]
            u_av = u_l[0, l]

            # Mean displacement of a classical oscillator at temperature T
            u_av *= sqrt(kT) / abs(omega)

            mode_av = np.zeros((len(self.atoms), 3), dtype=complex)
            # Insert slice with atomic displacements for the included atoms
            mode_av[self.indices] = u_av
            # Repeat and multiply by Bloch phase factor
            mode_Nav = np.vstack(N * [mode_av]) * phase_Na[:, np.newaxis]

            traj = Trajectory('%s.mode.%d.traj' % (self.name, l), 'w')

            for x in np.linspace(0, 2 * pi, nimages, endpoint=False):
                atoms.set_positions((pos_Nav + np.exp(1.j * x) *
                                     mode_Nav).real)
                traj.write(atoms)

            traj.close()
            
# ------------------------------------------------------------------------------
# internal functions
# ------------------------------------------------------------------------------

def _phonons_run_force1_rot(force, ws, invL, rot, symprec=1e-6):
    """From a given force set, we derive a rotated force set 
       by applying a single corresponding symmetry operator.
       
       code outrageously adapted from PHON/src/set_forces by D. Alfe
       
       input:
          force:    an array of Forces, size [natoms, 3]
          ws:       the Wigner-Seitz positions, supercell (Cartesian), obtained 
                    from _get_wigner_seitz(supercell)
          invL:     the inverse of the lattice supercell
          rot:      the rotation matrix
          symprec:  the precision for comparing positions
   
       output:
          nforce:   the rotated forces set
    """
    
    # this routine has been validated and behaves as in PHON/set_forces.f
    # the force is properly rotated when given the 'rot' matrix.
    
    # exit if no WS cell defined
    if ws is None:
        return None
        
    # we will now apply the rotation to the force array
    nforce = force.copy()

    # scan all atoms to compute the rotated force matrix
    # F_na ( S*u ) = S * F_{S^-1*na} ( u )
    for a in range(len(ws)):    # atom index in super cell
        # compute equivalent atom location from inverse rotation in WS cell 
        # temp = S^-1 * a

        # inverse rotate, then to fractional (the other way does not work...)
        # invL * rot * ws[a] == (invL * rot^-1 * L) * (invL * ws[a])
        temp = np.dot(invL.T, np.dot(np.linalg.inv(rot), ws[a]))
        
        # find nb so that b = S^-1 * a: only on the equivalent atoms 'rot'
        for b in range(len(ws)):
            found1 = False
            tmp1  = np.dot(invL.T, ws[b]) # to fractional
            # check that the fractional part is the same
            delta = temp - tmp1
            if np.linalg.norm(delta - np.rint(delta)) < symprec:
                #          ws[b] = rot^-1 * ws[a] : 'b' is equivalent to 'a' by 'rot'
                #    rot * ws[b] =          ws[a]
                #           F[b] = rot^-1 * F[a] 
                #    rot *  F[b] = F[a] 
                found1 = True
                # the rotation matrix rot is applied as is to
                # the force in PHON/set_forces.    
                nforce[a] = np.dot(rot, force[b])  # apply rotation
                break # for b
    
        if not found1:
            nforce = None
            break # for a
        # end for a
    
    return nforce
    
def _get_wigner_seitz(atoms, move=0, wrap=1, symprec=1e-6):
    """Compute the Wigner-Seitz cell.
       this routine is rather slow, but it allows to limit the number of moves
       
       code outrageously copied from PHON/src/get_wig by D. Alfe
       
       This is equivalent to a Voronoi cell, but centered on atom 'move'
       An alternative is to use scipy.spatial.Voronoi, but it does not properly
       center the cell. Slow but secure...
       
       input:
          atoms: an ASE Atoms object, supercell
          move:  the index  of the atom in the cell to be used as center
          wrap:  the extent of the supercell
       output:
          ws:    atom positions in the Wigner-Seitz cell, Cartesian coordinates
    """
    
    # this function is fully equivalent to the PHON/src/get_wig.f
    
    # get the cell definition
    L     = atoms.get_cell()
    # get fractional coordinates in the cell/supercell
    xtmp  = atoms.get_scaled_positions().copy()
    # set the origin on the 'moving' atom
    xtmp -= xtmp[move]

    # construct the WS cell (defined as the closest vectors to the origin)
    r = range(-wrap,wrap+1)
    for na in range(len(xtmp)):
        # convert from fractional to Cartesian
        xtmp[na] = np.dot(L.T, xtmp[na].T)
        temp1 = xtmp[na]
        for i in r:
            for j in r:
                for k in r:
                    temp = xtmp[na] + i*L[0] + j*L[1] + k*L[2]
                    if np.all(abs(np.linalg.norm(temp) < np.linalg.norm(temp1))):
                        temp1 = temp
        xtmp[na] = temp1
    
    return xtmp
    
def _move_is_independent(dxlist, dx, symprec=1e-6):
    """Test if a vector is independent compared to those in a list.
       The test is for collinearity, and singularity of the formed basis
       
       input:
           dxlist:  a list of vectors
           dx:      a new vector which is tested
           symprec: precision for comparing vectors
       output:
           True when dx is independent from dxlist, False otherwise.
    """

    # test if the rotated displacement is collinear to
    # a stored one (in list 'dxlist'). test is done on normalised vectors.
    if dx is None:
        return False
    
    dxnorm      = np.asarray([x/np.linalg.norm(x) for x in dxlist])
    iscollinear = False
    for index in range(len(dxnorm)):
        if np.linalg.norm(np.cross(dx/np.linalg.norm(dx), dxnorm[index])) < symprec:
            iscollinear = True
            break
    if iscollinear:
        return False  # collinear
            
    # Test for singular matrix when adding new move
    dxlist2 = dxlist[:] # copy the list
    dxlist2.append(dx)
    dxnorm = np.asarray([x/np.linalg.norm(x) for x in dxlist2])
    if len(dxlist2) == 3 and np.abs(np.linalg.det(dxnorm)) < symprec:
        return False # singular
        
    return True
    
# ------------------------------------------------------------------------------
def _rotated_displacements(disp, sg):
    """Check if a displacement can be imaged into other ones using the symmetry
       operators of the spacegroup.
       
       input:
           disp: an initial displacement (3-vector)
           sg:   ASE Space group
       output:
           dxlist:  list of independent displacements imaged from 'disp'
           rotlist: list of corresponding rotations from 'disp'
    """

    # we try all symmetry operations in the spacegroup
    dxlist = [disp]
    rotlist= [None]
    for rot, trans in sg.get_symop():
    
        # find the equivalent displacement from initial displacement
        # and rotation. First 'rot' is identity.
        dx = np.dot(rot, disp) # in Cartesian: dx = rot * disp
        
        # check if that rotated move contributes to an independent basis set
        if _phonons_move_is_independent(dxlist, dx):
            # store dx, rot; not work for disp itself that was added before
            dxlist.append(dx)
            rotlist.append(rot)
        
        # exit when 3 moves have been found from same generator move
        if len(dxlist) == 3:
            break
            
    return dxlist, rotlist
    
# ------------------------------------------------------------------------------
def _isfile_parallel(filename):
    """Check if a file is opened, taking into account the MPI environment
    
    input:
        filename
    output:
        0 when the file does not exist, 1 if it does
    """

    if world.rank == 0:
        isf = os.path.isfile(filename)
    else:
        isf = 0
    # Synchronize:
    return world.sum(isf)
