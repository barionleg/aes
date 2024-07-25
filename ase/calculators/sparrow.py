"""This module defines an ASE interface to
[sparrow](https://github.com/qcscine/sparrow)

Usage: (Tested only with Sparrow v3)

atoms = Atoms(...)
atoms.calc = Sparrow(charge=0, magmom=0)

"""

from ase.calculators.calculator import Calculator, all_changes
from ase.units import Bohr, Hartree

try:
    import scine_utilities as su
    import scine_sparrow
    _manager = su.core.ModuleManager()
    _prop_dict = {'energy': su.Property.Energy,
                  'forces': su.Property.Gradients,
                  'free_energy': su.Property.Energy}
    have_sparrow = True
except ImportError:
    _manager = None
    _prop_dict = {}
    have_sparrow = False


class Sparrow(Calculator):
    implemented_properties = ['energy', 'forces', 'free_energy']
    # TODO: consider 'stress', 'dipole', 'charges', 'magmom' and 'magmoms' props
    #implemented_properties += ['stress', 'stresses']  # bulk properties
    default_parameters = {
        'method': 'PM6',
        'charge': 0,
        'magmom': 0
    }
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        method: str
          Semi-empirical method employed (MNDO, AM1, RM1, PM3, PM6, DFTB0, DFTB2, DFTB3)

        charge: int
          Total charge (electron units)

        magmom: int
          Number of unpaired electrons.

        """
        if not have_sparrow:
            raise RuntimeError("Sparrow Python module could not be imported!")

        Calculator.__init__(self, **kwargs)
        self.calc = _manager.get('calculator', self.parameters.method)
        assert self.calc is not None
        self.calc.log.output.remove('cout')

    def _set_elements(self, symbols):
        elems = []
        for code in symbols:
            code = getattr(su.ElementType, code)
            elems.append(code)

        self.structure = su.AtomCollection(len(elems))
        self.structure.elements = elems

    def calculate(
        self,
        atoms=None,
        properties=None,
        system_changes=all_changes,
    ):
        # first call: 'positions', 'numbers', 'cell', 'pbc', 'initial_charges', 'initial_magmoms'
        # subsequent calls: just changed values
        #print(f"Called calculate(), changes={system_changes}, atoms = {atoms}")
        if properties is None:
            properties = self.implemented_properties
        Calculator.calculate(self, atoms, properties, system_changes)

        self.calc.set_required_properties(list(
            set(_prop_dict[p] for p in properties)))

        #natoms = len(self.atoms)

        if 'numbers' in system_changes:
            self._set_elements(atoms.symbols)
        if 'positions' in system_changes:
            self.structure.positions = self.atoms.positions / Bohr
            self.calc.structure = self.structure
        #if 'cell' in system_changes or 'pbc' in system_changes:
        #    cell = self.atoms.cell / Bohr  # TODO - use cell
        if 'initial_charges' in system_changes:
            self.calc.settings["molecular_charge"] = self.parameters.charge
        if 'initial_magmoms' in system_changes:
            magmom = self.parameters.magmom
            if magmom != 0:
                self.calc.settings["spin_mode"] = "unrestricted"
                self.calc.settings["spin_multiplicity"] = magmom+1

        ans = self.calc.calculate()

        if 'energy' in properties:
            self.results['energy'] = ans.energy * Hartree
        if 'free_energy' in properties:
            self.results['free_energy'] = ans.energy * Hartree
        if 'forces' in properties:
            self.results['forces'] = ans.gradients * (-Hartree/Bohr)
