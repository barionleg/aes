"""
ASE Calculator for interatomic models compatible with the Knowledgebase of
Interatomic Models (KIM) application programming interface (API). Written by:

Mingjian Wen
University of Minnesota
"""
from __future__ import absolute_import, division, print_function
from collections import defaultdict
import numpy as np
from .exceptions import KIMCalculatorError
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import equal
from ase import Atom
from ase.neighborlist import neighbor_list
try:
    import kimpy
    from kimpy import neighlist as nl
except ImportError:
    raise RuntimeError('kimpy not found; KIM calculator will not work')


__version__ = '0.2.0'
__author__ = 'Mingjian Wen'


class KIMModelCalculator(Calculator, object):
    """ An ASE calculator to work with KIM interatomic models.

    Parameter
    ---------

    modelname: str
      KIM model name

    ase_neigh: bool
      False: use the neighbor list library in kimpy
      True: use ase neighbor list (usually slower than the kimpy neighbor list library)

    neigh_skin_ratio: double
      The neighbor list is build using r_neigh = (1+neigh_skin_ratio)*rcut.

    release_GIL: bool
      Whether to release python GIL s.t. a KIM model can run with multiple threads

    debug: bool
      Flag to indicate whether to enable debug mode to print extra information.
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, modelname, ase_neigh=False, neigh_skin_ratio=0.2,
                 release_GIL=False, debug=False, *args, **kwargs):
        super(KIMModelCalculator, self).__init__(*args, **kwargs)

        self.modelname = modelname
        self.ase_neigh = ase_neigh
        self.release_GIL = release_GIL
        self.debug = debug

        # neigh attributes
        if neigh_skin_ratio < 0:
            neigh_skin_ratio = 0
        self.neigh_skin_ratio = neigh_skin_ratio
        self.neigh = None
        self.skin = None
        self.influence_dist = None
        self.cutoffs = None
        self.last_update_positions = None
        if self.ase_neigh:
            self.update_neigh = self.update_ase_neigh
        else:
            self.update_neigh = self.update_kimpy_neigh

        # padding atoms related
        self.padding_need_neigh = None
        self.num_contributing_particles = None
        self.num_padding_particles = None
        self.padding_image_of = None

        # model and compute arguments objects
        self.kim_model = None
        self.compute_args = None

        # model input
        self.num_particles = None
        self.species_code = None
        self.particle_contributing = None
        self.coords = None

        # model output
        self.energy = None
        self.forces = None

        # initialization flags
        self.kim_initialized = False
        self.neigh_initialized = False

        self.species_map = None

        self.init_kim()
        self.init_neigh()

    def init_kim(self):
        """Initialize KIM.
        """

        if self.kim_initialized:
            return

        # create model
        units_accepted, kim_model, error = kimpy.model.create(
            kimpy.numbering.zeroBased,
            kimpy.length_unit.A,
            kimpy.energy_unit.eV,
            kimpy.charge_unit.e,
            kimpy.temperature_unit.K,
            kimpy.time_unit.ps,
            self.modelname
        )
        check_error(error, 'kimpy.model.create')
        if not units_accepted:
            report_error('requested units not accepted in kimpy.model.create')
        self.kim_model = kim_model

        # units
        if self.debug:
            l_unit, e_unit, c_unit, te_unit, ti_unit = kim_model.get_units()
            check_error(error, 'kim_model.get_units')
            print('Length unit is:', str(l_unit))
            print('Energy unit is:', str(e_unit))
            print('Charge unit is:', str(c_unit))
            print('Temperature unit is:', str(te_unit))
            print('Time unit is:', str(ti_unit))
            print()

        # create compute arguments
        self.compute_args, error = kim_model.compute_arguments_create()
        check_error(error, 'kim_model.compute_arguments_create')

        # check compute arguments
        kimpy_arg_name = kimpy.compute_argument_name
        num_arguments = kimpy_arg_name.get_number_of_compute_argument_names()
        if self.debug:
            print('Number of compute_args:', num_arguments)

        for i in range(num_arguments):
            name, error = kimpy_arg_name.get_compute_argument_name(i)
            check_error(
                error, 'kimpy.compute_argument_name.get_compute_argument_name')

            dtype, error = kimpy_arg_name.get_compute_argument_data_type(name)
            check_error(error,
                        'kimpy.compute_argument_name.'
                        'get_compute_argument_data_type')

            arg_support, error = self.compute_args.get_argument_support_status(
                name)
            check_error(error, 'compute_args.get_argument_support_status')

            if self.debug:
                n_space_1 = 21 - len(str(name))
                n_space_2 = 7 - len(str(dtype))
                print('Compute Argument name "{}" '.format(name) + ' ' * n_space_1 +
                      'is of type "{}" '.format(dtype) + ' ' * n_space_2 +
                      'and has support status "{}".'.format(arg_support))

            # the simulator can handle energy and force from a kim model
            # virial is computed within the calculator
            if arg_support == kimpy.support_status.required:
                if (name != kimpy.compute_argument_name.partialEnergy and
                        name != kimpy.compute_argument_name.partialForces):
                    report_error(
                        'Unsupported required ComputeArgument {}'.format(name))

        # check compute callbacks
        callback_name = kimpy.compute_callback_name
        num_callbacks = callback_name.get_number_of_compute_callback_names()
        if self.debug:
            print()
            print('Number of callbacks:', num_callbacks)

        for i in range(num_callbacks):
            name, error = callback_name.get_compute_callback_name(i)
            check_error(error,
                        'kimpy.compute_callback_name'
                        '.get_compute_callback_name')

            callback_support = self.compute_args.get_callback_support_status
            support_status, error = callback_support(name)
            check_error(error, 'compute_args.get_callback_support_status')

            if self.debug:
                n_space = 18 - len(str(name))
                print('Compute callback "{}"'.format(name) + ' ' * n_space +
                      'has support status "{}".'.format(support_status))

            # cannot handle any "required" callbacks
            if support_status == kimpy.support_status.required:
                report_error(
                    'Unsupported required ComputeCallback: {}'.format(name))

        # set cutoff
        model_influence_dist = kim_model.get_influence_distance()
        self.skin = self.neigh_skin_ratio * model_influence_dist
        self.influence_dist = model_influence_dist + self.skin

        model_cutoffs, padding_not_require_neigh = kim_model.get_neighbor_list_cutoffs_and_hints()
        self.cutoffs = np.array(
            [cut + self.skin for cut in model_cutoffs], dtype=np.double)

        if padding_not_require_neigh.all():
            self.padding_need_neigh = False
        else:
            self.padding_need_neigh = True

        self.species_map = self.create_species_map()

        if self.debug:
            print()
            print('Model influence distance:', model_influence_dist)
            print('Calculator influence distance (include skin):',
                  self.influence_dist)
            print('Number of cutoffs:', model_cutoffs.size)
            print('Model cutoffs:', model_cutoffs)
            print('Model padding not require neighbors:',
                  padding_not_require_neigh)
            print('Calculator cutoffs (include skin):', self.cutoffs)
            print('Calculator cutoff skin:', self.skin)
            print()

        self.kim_initialized = True

    def init_neigh(self):
        """Initialize neighbor list.

        Parameter
        ---------

        atoms: ASE Atoms instance
        """
        # register get neigh callback
        if self.ase_neigh:
            neigh = {}
            self.neigh = neigh
            error = self.compute_args.set_callback(
                kimpy.compute_callback_name.GetNeighborList,
                get_neigh,
                neigh)
            check_error(error, 'compute_args.set_callback')
        else:
            neigh = nl.initialize()
            self.neigh = neigh
            error = self.compute_args.set_callback_pointer(
                kimpy.compute_callback_name.GetNeighborList,
                nl.get_neigh_kim(),
                neigh)

            check_error(error, 'compute_arguments.set_callback_pointer')

        self.neigh_initialized = True

    def build_neighbor_list(self, atoms):
        """Build the neighbor list and return an Atoms object with all
        the neighbors added.

        First a neighbor list is created from ase.neighbor_list, having
        only information about first neighbors of the original atoms
        if second neighbors are required they are calculated using information
        from the first neighbor list
        """
        syms = atoms.get_chemical_symbols()
        n = len(atoms)
        i, j, D, S, dists = neighbor_list('ijDSd', atoms, self.influence_dist)

        # Get coordinates for all neighbors (this has overlapping positions)
        A = atoms.get_positions()[i] + D

        # Make the neighbor list ready for KIM
        ac = atoms.copy()
        used = dict()

        # Variables below only include information for the neighbors (padding)
        padding_image_of = []
        neighbor_shifts = []

        # These variables include both original atoms and their neighbors
        neb_dict = defaultdict(list)
        neb_dists = defaultdict(list)

        # Loop over all neighbor pairs
        for k in range(len(i)):
            shift_tuple = tuple(S[k])
            t = (j[k], ) + shift_tuple
            if shift_tuple == (0, 0, 0):
                # In unit cell
                neb_dict[i[k]].append(j[k])
                neb_dists[i[k]].append(dists[k])
                if t not in used:
                    used[t] = j[k]
            else:
                # Not in unit cell
                if t not in used:
                    # Add the neighbor as a padding atom
                    used[t] = len(ac)
                    ac.append(Atom(syms[j[k]], position=A[k]))
                    padding_image_of.append(j[k])
                    neighbor_shifts.append(S[k])
                neb_dict[i[k]].append(used[t])
                neb_dists[i[k]].append(dists[k])
        neighbor_list_size = n

        # Add 2. neighbors if the potential requires them, i.e. information
        # of the padding atoms' neighbors
        if self.padding_need_neigh:
            neighbor_list_size = len(ac)
            inv_used = dict((v, k) for k, v in used.items())
            # Loop over all the neighbors (k)
            # and the image of that neighbor in the cell (neb)
            for k, neb in enumerate(padding_image_of[:]):
                # Shift from original atom in cell to neighbor
                shift = neighbor_shifts[k]
                for org_neb, org_dist in zip(neb_dict[neb], neb_dists[neb]):
                    # Get the shift of the neighbor of the original atom
                    org_shift = inv_used[org_neb][1:]

                    # Apply sum of original shift and current shift
                    # to neighbors of original atom
                    tot_shift = org_shift + shift

                    # Get the image in the cell of the original neighbor
                    if org_neb <= n - 1:
                        org_neb_image = org_neb
                    else:
                        org_neb_image = padding_image_of[org_neb - n]

                    # If the original image with the total shift has been
                    # used before then it is also a neighbor of this atom
                    tt = (org_neb_image, ) + tuple(tot_shift)
                    if tt in used:
                        neb_dict[k + n].append(used[tt])
                        neb_dists[k + n].append(org_dist)

        neb_lists = []
        for cut in self.cutoffs:
            neb_list = [np.array(neb_dict[k],
                                 dtype=np.intc)[neb_dists[k] <= cut]
                        for k in range(neighbor_list_size)]
            neb_lists.append(neb_list)

        self.padding_image_of = np.array(padding_image_of, dtype=np.intc)

        # neb_list now only contains neighbor information for the original
        # atoms. A neighbor is represented as an index in the list of all
        # coordinates in self.coords
        self.neigh['neighbors'] = neb_lists
        self.neigh['num_particles'] = neighbor_list_size

        return ac

    def update_ase_neigh(self, atoms):
        """Create the neighbor list along with the other required parameters.
        The required parameters are:
        - num_particles
        - coords
        - particle_contributing
        - species_code

        KIM requires a neighbor list that has indices corresponding to
        positions.
        """
        # Information of original atoms
        self.num_contributing_particles = len(atoms)

        ac = self.build_neighbor_list(atoms)

        # Save the number of atoms and all their neighbors and positions
        N = len(ac)
        num_padding = N - self.num_contributing_particles
        self.num_particles = np.array([N], dtype=np.intc)
        self.coords = ac.get_positions()

        # Save which coordinates are from original atoms and which are from
        # neighbors using a mask
        indices_mask = [1] * \
            self.num_contributing_particles + [0] * num_padding
        self.particle_contributing = np.array(indices_mask, dtype=np.intc)

        # species support and code
        all_species = ac.get_chemical_symbols()
        try:
            self.species_code = np.array([self.species_map[s] for s in all_species],
                                         dtype=np.intc)
        except KeyError as e:
            report_error('Species not support by KIM model; {}'.format(e))

        if self.debug:
            print('Debug: called update_ase_neigh')
            print()

    def update_kimpy_neigh(self, atoms):
        """Create neighbor list and model input.
        Parameter
        ---------

        atoms: ASE Atoms instance
        """

        # get info from Atoms object
        cell = np.asarray(atoms.get_cell(), dtype=np.double)
        pbc = np.asarray(atoms.get_pbc(), dtype=np.intc)
        contributing_coords = np.asarray(
            atoms.get_positions(), dtype=np.double)
        contributing_species = atoms.get_chemical_symbols()
        num_contributing = atoms.get_number_of_atoms()
        self.num_contributing_particles = num_contributing

        # species support and code
        try:
            contributing_species_code = np.array(
                [self.species_map[s] for s in contributing_species], dtype=np.intc)
        except KeyError as e:
            report_error('Species not support by KIM model; {}'.format(e))

        if pbc.any():  # need padding atoms
            # create padding atoms
            out = nl.create_paddings(self.influence_dist,
                                     cell,
                                     pbc,
                                     contributing_coords,
                                     contributing_species_code)
            padding_coords, padding_species_code, self.padding_image_of, error = out
            check_error(error, 'nl.create_paddings')
            num_padding = padding_species_code.size

            self.num_particles = np.array(
                [num_contributing + num_padding], dtype=np.intc)
            tmp = np.concatenate((contributing_coords, padding_coords))
            self.coords = np.asarray(tmp, dtype=np.double)
            tmp = np.concatenate(
                (contributing_species_code, padding_species_code))
            self.species_code = np.asarray(tmp, dtype=np.intc)
            self.particle_contributing = np.ones(
                self.num_particles[0], dtype=np.intc)
            self.particle_contributing[num_contributing:] = 0
            need_neigh = np.ones(self.num_particles[0], dtype=np.intc)
            if not self.padding_need_neigh:
                need_neigh[num_contributing:] = 0

        else:  # do not need padding atoms
            self.padding_image_of = np.array([])
            self.num_particles = np.array([num_contributing], dtype=np.intc)
            self.coords = np.array(contributing_coords, dtype=np.double)
            self.species_code = np.array(
                contributing_species_code, dtype=np.intc)
            self.particle_contributing = np.ones(
                num_contributing, dtype=np.intc)
            need_neigh = self.particle_contributing

        # create neighborlist
        error = nl.build(self.neigh, self.coords,
                         self.influence_dist, self.cutoffs, need_neigh)
        check_error(error, 'nl.build')

        if self.debug:
            print('Debug: called update_kimpy_neigh')
            print()

    def update_kim(self):
        """ Register model input and output data pointers.
        """
        # model output
        self.energy = np.array([0.], dtype=np.double)
        self.forces = np.zeros([self.num_particles[0], 3], dtype=np.double)

        # register argument
        error = self.compute_args.set_argument_pointer(
            kimpy.compute_argument_name.numberOfParticles, self.num_particles)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        error = self.compute_args.set_argument_pointer(
            kimpy.compute_argument_name.particleSpeciesCodes, self.species_code)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        error = self.compute_args.set_argument_pointer(
            kimpy.compute_argument_name.particleContributing, self.particle_contributing)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        error = self.compute_args.set_argument_pointer(
            kimpy.compute_argument_name.coordinates, self.coords)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        error = self.compute_args.set_argument_pointer(
            kimpy.compute_argument_name.partialEnergy, self.energy)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        error = self.compute_args.set_argument_pointer(
            kimpy.compute_argument_name.partialForces, self.forces)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        if self.debug:
            print('Debug: called update_kim')
            print()

    def update_kim_coords(self, atoms):
        """Update the atom positions in self.coords, which is registered in KIM.

        Parameter
        ---------

        atoms: ASE Atoms instance
        """
        if self.padding_image_of.size != 0:
            # displacement of contributing atoms
            disp_contrib = atoms.positions - self.last_update_positions
            # displacement of padding atoms
            disp_pad = disp_contrib[self.padding_image_of]
            # displacement of all atoms
            disp = np.concatenate((disp_contrib, disp_pad))
            # update coords in KIM
            self.coords += disp
        else:
            np.copyto(self.coords, atoms.positions)

        if self.debug:
            print('Debug: called update_kim_coords')
            print()

    def calculate(self, atoms=None,
                  properties=['energy', 'forces', 'stress'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc']):
        """
        Inherited method from the ase Calculator class that is called by get_property().

        Parameters
        ----------

        atoms: ASE Atoms instance

        properties: list of str
          List of what needs to be calculated.  Can be any combination
          of 'energy', 'forces' and 'stress'.

        system_changes: list of str
          List of what has changed since last calculation.  Can be
          any combination of these six: 'positions', 'numbers', 'cell',
          and 'pbc'.
        """

        Calculator.calculate(self, atoms, properties, system_changes)

        need_update_neigh = True
        if len(system_changes) == 1 and 'positions' in system_changes:
            # only position changes
            if self.last_update_positions is not None:
                a = self.last_update_positions
                b = atoms.positions
                if a.shape == b.shape:
                    delta = np.linalg.norm(a - b, axis=1)
                    # indices of the two largest elements
                    ind = np.argpartition(delta, -2)[-2:]
                    if sum(delta[ind]) <= self.skin:
                        need_update_neigh = False

        # update KIM API input data and neighbor list if necessary
        if system_changes:
            if need_update_neigh:
                self.update_neigh(atoms)
                self.update_kim()
                self.last_update_positions = atoms.get_positions()
            else:
                self.update_kim_coords(atoms)

            error = self.kim_model.compute(self.compute_args, self.release_GIL)
            check_error(error, 'kim_model.compute')

        energy = self.energy[0]
        forces = assemble_padding_forces(self.forces,
                                         self.num_contributing_particles,
                                         self.padding_image_of)
        stress = compute_virial_stress(self.forces, self.coords)

        # return values
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress

    def get_kim_model_supported_species(self):
        """Get all the supported species by the KIM model.

        Returns
        -------

        species: list of str
            a list of chemical symbols (e.g. ["Mo", "S"])
        """
        species = []
        num_kim_species = kimpy.species_name.get_number_of_species_names()

        for i in range(num_kim_species):
            species_name, error = kimpy.species_name.get_species_name(i)
            check_error(error, 'kimpy.species_name.get_species_name')
            species_support, code, error = self.kim_model.get_species_support_and_code(
                species_name)
            check_error(error, 'kim_model.get_species_support_and_code')
            if species_support:
                species.append(str(species_name))

        return species

    def create_species_map(self):
        """Get all the supported species of the KIM model and the corresponding codes.

        Returns
        -------

        species_map: dict
            key: str
                chemical symbols (e.g. "Ar")
            value: int
                species integer code (e.g. 1)
        """
        supported_species = self.get_kim_model_supported_species()
        species_map = dict()
        for s in supported_species:
            species_support, code, error = self.kim_model.get_species_support_and_code(
                kimpy.species_name.SpeciesName(s))
            check_error(error or not species_support,
                        'kim_model.get_species_support_and_code')
            species_map[s] = code
            if self.debug:
                print('Species {} is supported and its code is: {}'.format(s, code))
        return species_map

    def check_state(self, atoms, tol=1e-15):
        return compare_atoms(self.atoms, atoms)

    def __expr__(self):
        return 'KIMModelCalculator(modelname = {})'.format(self.modelname)

    def __del__(self):
        """Garbage collection for the KIM model object and neighbor list object."""

        if self.neigh_initialized:
            if self.ase_neigh:
                self.neigh = {}
            else:
                nl.clean(self.neigh)

        if self.kim_initialized:
            error = self.kim_model.compute_arguments_destroy(
                self.compute_args)
            check_error(error, 'kim_model.compute_arguments_destroy')

            kimpy.model.destroy(self.kim_model)


def compare_atoms(atoms1, atoms2, tol=1e-15):
    """Check for system changes since last calculation.
    Since each calculate call will compute all properties, we will just
    return the first difference to trigger a calculation.
    It uses the default tolerance in the used numpy functions.
    """
    if atoms1 is None:
        return ['positions', 'numbers', 'cell', 'pbc']
    else:
        system_changes = []
        if not equal(atoms1.positions, atoms2.positions, tol):
            system_changes.append('positions')
        if not equal(atoms1.numbers, atoms2.numbers):
            system_changes.append('numbers')
        if not equal(atoms1.cell, atoms2.cell, tol):
            system_changes.append('cell')
        if not equal(atoms1.pbc, atoms2.pbc):
            system_changes.append('pbc')

    return system_changes


def assemble_padding_forces(forces, n, padding_image_of):
    """
    Assemble forces on padding atoms back to contributing atoms.

    Parameters
    ----------

    forces: 2D array
      forces on both contributing and padding atoms

    n: int
      number of contributing atoms

    padding_image_of: 1D int array
      atom number, of which the padding atom is an image


    Returns
    -------
      Total forces on contributing atoms.
    """
    total_forces = np.array(forces[:n])

    has_padding = True if padding_image_of.size != 0 else False

    if has_padding:
        pad_forces = forces[n:]
        for f, org_index in zip(pad_forces, padding_image_of):
            total_forces[org_index] += f

    return total_forces


def compute_virial_stress(forces, coords):
    """Compute the virial stress in voigt notation.

    Parameters
    ----------
      forces: 2D array
        forces on all atoms (padding included)

      coords: 2D array
        coordinates of all atoms (padding included)

    Returns
    -------
      stress: 1D array
        stress in Voigt order (xx, yy, zz, yz, xz, xy)
    """
    stress = np.zeros(6)
    stress[0] = -np.dot(forces[:, 0], coords[:, 0])
    stress[1] = -np.dot(forces[:, 1], coords[:, 1])
    stress[2] = -np.dot(forces[:, 2], coords[:, 2])
    stress[3] = -np.dot(forces[:, 1], coords[:, 2])
    stress[4] = -np.dot(forces[:, 0], coords[:, 2])
    stress[5] = -np.dot(forces[:, 0], coords[:, 1])

    return stress


def check_error(error, msg):
    if error != 0 and error is not None:
        raise KIMCalculatorError('Calling "{}" failed.\n'.format(msg))


def report_error(msg):
    raise KIMCalculatorError(msg)


def get_neigh(data, cutoffs, neighbor_list_index, particle_number):
    try:
        # We can only return neighbors of particles that were stored
        number_of_particles = data['num_particles']
        if particle_number >= number_of_particles or particle_number < 0:
            return(np.array([]), 1)

        neighbors = data['neighbors'][neighbor_list_index][particle_number]
        return (neighbors, 0)

    except:
        return(np.array([]), 1)
