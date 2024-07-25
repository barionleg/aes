
import pynnp
import numpy as np
from ase.data import chemical_symbols, atomic_numbers
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)
from ase import units 


def elementmap_from_element_list(elements):
    nnp_element_map = pynnp.ElementMap()
    #nnp_structure = pynnp.Structure()
    element_line = ''
    for el in elements:
        element_line += el+' '
    nnp_element_map.registerElements(element_line)
    return nnp_element_map


def make_structure_from_ase_atoms(ase_atoms, elementMap, model_length_units=1.0):
    ## pynnp needs the atoms to be wrapped into the unit cell
    wrapped_atoms = ase_atoms.copy()
    wrapped_atoms.wrap()
    
    nnp_structure = pynnp.Structure()
    nnp_structure.setElementMap(elementMap)

    for ase_atom in wrapped_atoms:
        nnp_atom = pynnp.Atom()
        nnp_atom.r[0] = ase_atom.x/model_length_units
        nnp_atom.r[1] = ase_atom.y/model_length_units
        nnp_atom.r[2] = ase_atom.z/model_length_units
        nnp_structure.addAtom(nnp_atom, ase_atom.symbol)
            
    nnp_structure.isPeriodic = wrapped_atoms.pbc[0] # I should check all of the directions eventually
    nnp_structure.isTriclinic = True # does it matter if this is always true?
    
    for i in range(3):
        for j in range(3):
            nnp_structure.box[i][j] = wrapped_atoms.cell[i][j]/model_length_units

    return nnp_structure


def setup_nnp_mode(input_file, scaling_file, weight_file_format, 
    use_unscaled_symmetry_functions = False):

    # Initialize NNP setup (symmetry functions only).
    nnp_mode = pynnp.Mode()
    nnp_mode.initialize()
    # I think setupGeneric is only for unit testing
    #if hasattr(nnp_mode, 'setupGeneric'):  
    #    nnp_mode.setupGeneric()
    nnp_mode.loadSettingsFile(input_file)
    nnp_mode.setupNormalization()

    nnp_mode.setupElementMap()
    nnp_mode.setupElements()
    nnp_mode.setupCutoff()
    
    # each step prints to stdout a bunch of info, this one dumps the most.
    nnp_mode.setupSymmetryFunctions()
    # not all versions have these features
    if hasattr(nnp_mode, 'setupSymmetryFunctionMemory'):  
        nnp_mode.setupSymmetryFunctionMemory(verbose=False)
    if hasattr(nnp_mode, 'setupSymmetryFunctionCache'): 
        nnp_mode.setupSymmetryFunctionCache(verbose=False)
    nnp_mode.setupSymmetryFunctionGroups()
    
    if use_unscaled_symmetry_functions:
        nnp_mode.setupSymmetryFunctionScalingNone()
    else:
        nnp_mode.setupSymmetryFunctionScaling(scaling_file)
        # these are flags for warnings, what should we do? False supresses them
        #nnp_mode.setupSymmetryFunctionStatistics(True, True, True, False)
        nnp_mode.setupSymmetryFunctionStatistics(False, False, False, False) 

    nnp_mode.setupNeuralNetwork()
    nnp_mode.setupNeuralNetworkWeights(weight_file_format)

    return nnp_mode




def calculate_on_nnp_structure(nnp_str, nnp_mode):


    ### we need someone to try a network with reference energy offsets to ensure this is workint  
    nnp_mode.removeEnergyOffset(nnp_str);

    use_normalization = False
    ### we need someone to do a normalization check to ensure this block works
    if hasattr(nnp_mode, 'useNormalization'): # this isn't always an available attribute 
        use_normalization = nnp_mode.useNormalization()
        # If normalization is used, convert structure data.
        if use_normalization:
            # these might not be implemented in all versions, hasattr might be needed in the future
            meanEnergy = nnp_mode.getMeanEnergy()
            convEnergy = nnp_mode.getConvEnergy()
            convLength = nnp_mode.getConvLength()
            nnp_str.toNormalizedUnits(meanEnergy, convEnergy, convLength)
    
    # Retrieve cutoff radius form NNP setup.
    cutoffRadius = nnp_mode.getMaxCutoffRadius()
    #print("Cutoff radius = ", cutoffRadius/convLength)

    # Calculate neighbor list.
    nnp_str.calculateNeighborList(cutoffRadius)

    # Calculate symmetry functions for all atoms (use groups).
    #nnp_mode.calculateSymmetryFunctions(nnp_str, True)
    nnp_mode.calculateSymmetryFunctionGroups(nnp_str, derivatives = True)

    # Calculate atomic neural networks.
    nnp_mode.calculateAtomicNeuralNetworks(nnp_str, derivatives = True)

    # Sum up potential energy.
    nnp_mode.calculateEnergy(nnp_str)

    # Collect force contributions.
    nnp_mode.calculateForces(nnp_str)

    if use_normalization:
        nnp_str.toPhysicalUnits(meanEnergy, convEnergy, convLength)

    ### I think this is important to do after denormalization
    nnp_mode.addEnergyOffset(nnp_str, False);
    nnp_mode.addEnergyOffset(nnp_str, True);

    if False: # These should be documented as notes for future developers. 
        print("dir(nnp_str):")
        for meth in dir(nnp_str):
            print(meth)

        print("dir(nnp_mode):")
        for meth in dir(nnp_mode):
            print(meth)

        print("dir(nnp_str.atoms[0]):")
        for meth in dir(nnp_str.atoms[0]):
            print(meth)

        print("------------")
        print("Structure:")
        print("------------")
        print("Cell:") 
        print('a]',nnp_str.box[0])
        print('b]',nnp_str.box[1])
        print('c]',nnp_str.box[2])
        print("numAtoms           : ", nnp_str.numAtoms)
        print("numAtomsPerElement : ", nnp_str.numAtomsPerElement)
        print("------------")
        print("Energy (Ref) : ", nnp_str.energyRef)
        print("Energy (NNP) : ", nnp_str.energy)
        print("------------")
        for atom in nnp_str.atoms:
            print(atom.index, atom.element, nnp_mode.elementMap[atom.element], atom.energy, atom.f.r)

from .n2p2 import parse_input_file

class PyNNP(Calculator):
    '''uses to n2p2's python interface'''

    implemented_properties = ['energy', 'free_energy', 'energies', 'forces']
    #, 'stress', 'stresses']
    #                              'stress', 'magmom', 'magmoms']

    nolabel = True

    #default_parameters = {'asap_cutoff': False}
    

    def __init__(self, 
                input_file         = 'input.nn',
                weight_file_format = 'weights.%03zu.data',
                scaling_file       = 'scaling.data',
                use_unscaled_symmetry_functions = False,
                atoms=None,
                model_length_units = units.Ang, #model units to ASE units, i.e. units.Bohr
                model_energy_units = units.eV,
                **kwargs,
                ):
        
        self.elements = None
        self.input_file = input_file
        self.weight_file_format = weight_file_format
        self.scaling_file       = scaling_file
        self.use_unscaled_symmetry_functions = use_unscaled_symmetry_functions
        self.model_length_units = model_length_units
        self.model_energy_units = model_energy_units
        self.nnp_mode = None
        self.G = None
        self.dGdr = None
        self.dEdG = None
        self.input_parameters = {}
        self.input_symmetry_functions = []
        Calculator.__init__(self, **kwargs)

    def initialize(self):#, atoms):
        self.input_parameters, self.input_symmetry_functions = parse_input_file(self.input_file)
        self.elements = self.input_parameters['elements']
    
        self.elementmap = elementmap_from_element_list(self.elements)
        self.nnp_mode = setup_nnp_mode( self.input_file, self.scaling_file, self.weight_file_format,
                                        self.use_unscaled_symmetry_functions) 

    
    def calculate(self, atoms=None, properties=['energy'],
              system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        #if 'numbers' in system_changes or self.nnp_mode is None:
        #    self.initialize(self.atoms)
        if self.nnp_mode is None:
             self.initialize()#self.atoms)
        
        nnp_str = make_structure_from_ase_atoms(atoms, self.elementmap, self.model_length_units)
        calculate_on_nnp_structure(nnp_str, self.nnp_mode)
        
        self.results['energy'] = nnp_str.energy*self.model_energy_units
        self.results['free_energy'] = nnp_str.energy*self.model_energy_units
        energies=np.zeros(len(atoms))
        forces  =np.zeros((len(atoms),3))
        self.G = []
        self.dGdR = []
        self.dEdG = []
        for i in range(len(atoms)):
            energies[i] = nnp_str.atoms[i].energy 
            forces[i]    = nnp_str.atoms[i].f.r
            self.G.append( np.array(nnp_str.atoms[i].G) )
            self.dEdG.append( np.array(nnp_str.atoms[i].dEdG)*self.model_energy_units )
            # atoms[i].dGdr is a list of pynnp.Vec3D like  <pynnp.Vec3D object at 0x7fafba834c40>
            # this converts to np.arrays
            atom_dGdR       = nnp_str.atoms[i].dGdr 
            atom_dGdR_array = np.zeros((len(atom_dGdR),3))
            for i in range(len(atom_dGdR)):
                for j in range(3):
                    atom_dGdR_array[i,j]=atom_dGdR[i][j]
            self.dGdR.append( atom_dGdR_array / self.model_length_units) 
            

        
        self.results['energies'] = energies*self.model_energy_units
        self.results['forces'] = forces * self.model_energy_units/self.model_length_units
        
        if 'stress' in properties:
            print('Try the numeric force wrapper?')
            raise PropertyNotImplementedError
            


