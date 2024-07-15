"""Modules for calculating thermochemical information from computational
outputs."""

import os
import sys
from warnings import warn
from typing import Dict, List, Tuple, Literal
from numbers import Real
from abc import ABC

import numpy as np

from ase import units


class ThermoChem(ABC):
    """Abstract base class containing common methods used in thermochemistry
    calculations."""

    @property
    def imag_modes_handling(self):
        """For backwards compatibility, delete the following after some time."""
        return self._imag_modes_handling
    
    @imag_modes_handling.setter
    def imag_modes_handling(self, value):
        """For backwards compatibility, raise a deprecation warning."""
        warn("The imag_modes_handling attribute is deprecated and will be removed in a future release. "
             "Please use the raise_to attribute instead.", DeprecationWarning)
        self._imag_modes_handling = value


    @property
    def frequencies(self):
        """Returns the vibrational frequencies in m^-1."""
        return np.array(self.vib_energies) / units._hplanck  * units._e 
    
    def get_ZPE_correction(self):
        """Returns the zero-point vibrational energy correction in eV."""
        return 0.5 * np.sum(self.vib_energies)

    @staticmethod
    def get_ideal_translational_energy(temperature) -> float:
        """Returns the translational heat capacity times T in eV.
        
        Parameters
        ----------
        temperature : float
            The temperature in Kelvin.

        Returns
        -------
        float
        """
        return 3. / 2. * units.kB * temperature  # translational heat capacity (3-d gas)

    @staticmethod
    def get_ideal_rotational_energy(geometry: Literal["nonlinear", "linear", "monatomic"],
                                    temperature: Real) -> float:
        """Returns the rotational heat capacity times T in eV.
        
        Parameters
        ----------
        geometry : str
            The geometry of the molecule. Options are 'nonlinear', 'linear', and 'monatomic'.
        temperature : float
            The temperature in Kelvin.
        
        Returns
        -------
        float
        """
        if geometry == 'nonlinear':  # rotational heat capacity
            Cv_r = 3. / 2. * units.kB
        elif geometry == 'linear':
            Cv_r = units.kB
        elif geometry == 'monatomic':
            Cv_r = 0.
        else:
            raise ValueError('Invalid geometry: %s' % geometry)
        return Cv_r * temperature
    
    def get_ideal_trans_entropy(self, atoms, temperature) -> float:
        """Returns the translational entropy in eV/K.
        
        Parameters
        ----------
        atoms : ase.Atoms
            The atoms object.
        temperature : float
            The temperature in Kelvin.
        
        Returns
        -------
        float
        """
        # Translational entropy (term inside the log is in SI units).
        mass = sum(self.atoms.get_masses()) * units._amu  # kg/molecule
        S_t = (2 * np.pi * mass * units._k *
               temperature / units._hplanck**2)**(3.0 / 2)
        S_t *= units._k * temperature / self.referencepressure
        S_t = units.kB * (np.log(S_t) + 5.0 / 2.0)
        return S_t


    def get_vib_energy_contribution(self, temperature) -> float:
        """Calculates the change in internal energy due to vibrations from
        0K to the specified temperature for a set of vibrations given in
        eV and a temperature given in Kelvin.
        Returns the energy change in eV."""
        kT = units.kB * temperature
        dU = 0.

        for energy in self.vib_energies:
            dU += energy / (np.exp(energy / kT) - 1.)
        return dU

    def get_vib_entropy_contribution(self, temperature, return_list=False):
        """Calculates the entropy due to vibrations for a set of vibrations
        given in eV and a temperature given in Kelvin.  Returns the entropy
        in eV/K."""
        kT = units.kB * temperature
        S_v = 0.
        energies = np.array(self.vib_energies)
        energies /= kT # eV/ eV/K*K
        S_v = energies / (np.exp(energies) - 1.) - np.log(1. - np.exp(-energies))
        S_v *= units.kB
        if return_list:
            return S_v
        else:
            return np.sum(S_v)

    def _vprint(self, text):
        """Print output if verbose flag True."""
        if self.verbose:
            sys.stdout.write(text + os.linesep)

    def _head_gordon_damp(self, freq) -> float:
        """Head-Gordon damping function.

        Equation 8 from :doi:`10.1002/chem.201200497`"""
        ret = 1 / (1 + (self.tau_freq / freq)**self.alpha)

        return ret

    def get_qRRHO_entropy_contribution(self, temperature) -> float:
        """qRRHO Vibrational Entropy Contribution from
        :doi:`10.1002/chem.201200497`.

        Uses tau_freq to switch between S_v and S_r.

        Returns the entropy in eV/K."""
        qRRHO_vib = self.get_vib_entropy_contribution(temperature, return_list=True)
        qRRHO_rot = self.calc_qRRHO_entropies_r(temperature)
        assert len(qRRHO_vib) == len(qRRHO_rot), "qRRHO_vib and qRRHO_rot do not match"
        assert len(qRRHO_vib) == len(self.vib_energies), "qRRHO and frequencies do not match"
        S = 0.
        #simply use a cutoff to switch between S_v and S_r
        for n, freq in enumerate(self.frequencies):
            if freq < self.tau_freq:
                S += qRRHO_rot[n]
            else:
                S += qRRHO_vib[n]
        return S

    def calc_qRRHO_entropies_r(self, temperature) -> np.array:
        """Calculates the rotation of a rigid rotor for low frequency modes.

        Equation numbering from :doi:`10.1002/chem.201200497`

        Returns the entropy contribution in eV/K."""
        # rotational entropy
        S_r_damp = 0.
        kT = units._k * temperature
        R = units._k * units._Nav
        inertias = (self.atoms.get_moments_of_inertia() * units._amu /
                    (1e10**2))  # from amu/A^2 to kg m^2
        B_av = np.mean(inertias)

        # eq 4
        mu = units._hplanck / (8 * np.pi**2 * self.frequencies)
        # eq 5
        mu_prime = (mu * B_av) / (mu + B_av)  # kg m^2
        # eq 6
        x = np.sqrt(8 * np.pi**3 * mu_prime * kT / (units._hplanck)**2)
        # filter zeros out and set them to zero
        log_x = np.log(x, out=np.zeros_like(x, dtype='float64'), where=(x != 0))
        S_r_components = R * (1 / 2 + log_x)  # J/(Js)^2
        S_r_components *= units.J / units._Nav
        return S_r_components


    def get_msRRHO_vib_entropy_v_contribution(self, temperature) -> float:
        """msRRHO Vibrational Entropy Contribution from
        :doi:`10.1039/D1SC00621E`.

        Returns the entropy contribution in eV/K."""
        
        qRRHO = self.get_vib_entropy_contribution(temperature, return_list=True)
        assert len(qRRHO) == len(self.vib_energies), "qRRHO and frequencies do not match"
        return (self._head_gordon_damp(self.frequencies) * qRRHO).sum()
    
    def get_msRRHO_vib_entropy_r_contribution(self, temperature) -> float:
        """Calculates the rotation of a rigid rotor for low frequency modes.

        Equation numbering from :doi:`10.1002/chem.201200497`

        Returns the entropy contribution in eV/K."""
        S_r_components = self.calc_qRRHO_entropies_r(temperature)
        assert len(S_r_components) == len(self.vib_energies), "qRRHO and frequencies do not match"
        # eq 7
        return ((1 - self._head_gordon_damp(self.frequencies)) * S_r_components).sum()
    
    def get_damped_vib_energy_contribution(self, temperature):
        assert False, "Not implemented yet"
        H_v_damp = 0.
        R = units._k * units._Nav

        for n, freq in enumerate(self.frequencies):
            x = units._hplanck * freq / units._k
            comp = R * x * ( 0.5 + 1 / ( np.exp(x / temperature) - 1 ) )
            H_v_damp += self._head_gordon_damp(freq) * comp
        return H_v_damp
    
    def get_damped_free_rotor_energy_contribution(self, temperature):
        assert False, "Not implemented yet"
        H_r_damp = 0.
        R = units._k * units._Nav
        for n, freq in enumerate(self.frequencies):
            comp = 0.5 * R * temperature
            H_r_damp += 1 - self._head_gordon_damp(freq) * comp
        return H_r_damp
    
    def get_ideal_entropy(self, temperature, translation=False,
                    vibration=False, rotation=False, geometry=None,
                    electronic=False, pressure=None) ->Tuple[float, Dict[str, float]]:
        """Returns the entropy, in eV/K and a dict of the contributions"""
        S = 0.0
        ret = {}

        if translation:
            S_t = self.get_ideal_trans_entropy(self.atoms, temperature)
            ret['S_t'] = S_t
            S += S_t
            if pressure:
                # Pressure correction to translational entropy.
                S_p = - units.kB * np.log(pressure / self.referencepressure)
                S += S_p
                ret['S_p'] = S_p

        if rotation:
            # Rotational entropy (term inside the log is in SI units).
            if geometry == 'monatomic':
                S_r = 0.0
            elif geometry == 'nonlinear':
                inertias = (self.atoms.get_moments_of_inertia() * units._amu /
                            1e10**2)  # kg m^2
                S_r = np.sqrt(np.pi * np.prod(inertias)) / self.sigma
                S_r *= (8.0 * np.pi**2 * units._k * temperature /
                        units._hplanck**2)**(3.0 / 2.0)
                S_r = units.kB * (np.log(S_r) + 3.0 / 2.0)
            elif geometry == 'linear':
                inertias = (self.atoms.get_moments_of_inertia() * units._amu /
                            (10.0**10)**2)  # kg m^2
                inertia = max(inertias)  # should be two identical and one zero
                S_r = (8 * np.pi**2 * inertia * units._k * temperature /
                    self.sigma / units._hplanck**2)
                S_r = units.kB * (np.log(S_r) + 1.)
            S += S_r
            ret['S_r'] = S_r
        

        # Electronic entropy.
        if electronic:
            S_e = units.kB * np.log(2 * self.spin + 1)
            S += S_e
            ret['S_e'] = S_e

        # Vibrational entropy
        if vibration:
            S_v = self.get_vib_entropy_contribution(temperature)
            S += S_v
            ret['S_v'] = S_v

        return S, ret


    def _raise(self, raise_to) -> List[float]:
        return [raise_to if x < raise_to else x for x in self.vib_energies]


class HarmonicThermo(ThermoChem):
    """Class for calculating thermodynamic properties in the approximation
    that all degrees of freedom are treated harmonically. Often used for
    adsorbates.

    Inputs:

    vib_energies : list
        a list of the harmonic energies of the adsorbate (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of
        energies should match the number of degrees of freedom of the
        adsorbate; i.e., 3*n, where n is the number of atoms. Note that
        this class does not check that the user has supplied the correct
        number of energies. Units of energies are eV.
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    imag_modes_handling : string
        If 'remove', any imaginary frequencies will be removed in the
        calculation of the thermochemical properties.
        If 'error' (default), an error will be raised if any imaginary
        frequencies are present.
        If 'invert', the imaginary frequencies will be multiplied by -i.
    """

    def __init__(self, vib_energies, potentialenergy=0.,
                 imag_modes_handling='error'):

        # Check for imaginary frequencies.
        vib_energies, n_imag = _clean_vib_energies(
            vib_energies, handling=imag_modes_handling
        )
        self.vib_energies = vib_energies
        self.n_imag = n_imag

        self.potentialenergy = potentialenergy

    def get_internal_energy(self, temperature, verbose=True):
        """Returns the internal energy, in eV, in the harmonic approximation
        at a specified temperature (K)."""

        self.verbose = verbose
        vprint = self._vprint
        fmt = '%-15s%13.3f eV'
        vprint('Internal energy components at T = %.2f K:' % temperature)
        vprint('=' * 31)

        U = 0.

        vprint(fmt % ('E_pot', self.potentialenergy))
        U += self.potentialenergy

        zpe = self.get_ZPE_correction()
        vprint(fmt % ('E_ZPE', zpe))
        U += zpe

        dU_v = self.get_vib_energy_contribution(temperature)
        vprint(fmt % ('Cv_harm (0->T)', dU_v))
        U += dU_v

        vprint('-' * 31)
        vprint(fmt % ('U', U))
        vprint('=' * 31)
        return U

    def get_entropy(self, temperature, verbose=True):
        """Returns the entropy, in eV/K, in the harmonic approximation
        at a specified temperature (K)."""

        self.verbose = verbose
        vprint = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        vprint('Entropy components at T = %.2f K:' % temperature)
        vprint('=' * 49)
        vprint('%15s%13s     %13s' % ('', 'S', 'T*S'))

        S = 0.

        S_v = self.get_vib_entropy_contribution(temperature)
        vprint(fmt % ('S_harm', S_v, S_v * temperature))
        S += S_v

        vprint('-' * 49)
        vprint(fmt % ('S', S, S * temperature))
        vprint('=' * 49)
        return S

    def get_helmholtz_energy(self, temperature, verbose=True):
        """Returns the Helmholtz free energy, in eV, in the harmonic
        approximation at a specified temperature (K)."""

        self.verbose = True
        vprint = self._vprint

        U = self.get_internal_energy(temperature, verbose=verbose)
        vprint('')
        S = self.get_entropy(temperature, verbose=verbose)
        F = U - temperature * S

        vprint('')
        vprint('Free energy components at T = %.2f K:' % temperature)
        vprint('=' * 23)
        fmt = '%5s%15.3f eV'
        vprint(fmt % ('U', U))
        vprint(fmt % ('-T*S', -temperature * S))
        vprint('-' * 23)
        vprint(fmt % ('F', F))
        vprint('=' * 23)
        return F


class QuasiHarmonicThermo(ThermoChem):
    """Subclass of :class:`ThermoChem`, including the quasi-harmonic
    approximation of Cramer, Truhlar and coworkers :doi:`10.1021/jp205508z`.

    Inputs:

    vib_energies : list
        a list of the harmonic energies of the adsorbate (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of
        energies should match the number of degrees of freedom of the
        adsorbate; i.e., 3*n, where n is the number of atoms. Note that
        this class does not check that the user has supplied the correct
        number of energies. Units of energies are eV.
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    imag_modes_handling : string
        If 'remove', any imaginary frequencies will be removed in the
        calculation of the thermochemical properties.
        If 'error', an error will be raised if any imaginary
        frequencies are present.
        If 'invert', the imaginary frequencies will be multiplied by -i.
        If 'raise' (default), the imaginary frequencies will be raised to a
        certain value, specified by the *raise_to* keyword.
    raise_to : float
        The value to which imaginary frequencies will be raised if
        *imag_modes_handling* is 'raise'. Defaults to :math:`100cm^{-1}`.

    """

    def __init__(self, vib_energies, potentialenergy=0.,
                 imag_modes_handling='raise',
                 raise_to=100 * units.invcm, **kwargs) -> None:

        # Raise all imaginary frequencies
        vib_energies, n_imag = _clean_vib_energies(
            vib_energies, handling=imag_modes_handling,
            value=raise_to
        )
        self.vib_energies = vib_energies
        self.n_imag = n_imag
        # raise the low frequencies to a certain value
        self.vib_energies = self._raise(raise_to)
        self.potentialenergy = potentialenergy


    def get_internal_energy(self, temperature, verbose=True):
        """Returns the internal energy, in eV, in the harmonic approximation
        at a specified temperature (K)."""
        #copy paste of HarmonicThermo.get_internal_energy at the moment
        self.verbose = verbose
        vprint = self._vprint
        fmt = '%-15s%13.3f eV'
        vprint('Internal energy components at T = %.2f K:' % temperature)
        vprint('=' * 31)

        U = 0.

        vprint(fmt % ('E_pot', self.potentialenergy))
        U += self.potentialenergy

        zpe = self.get_ZPE_correction()
        vprint(fmt % ('E_ZPE', zpe))
        U += zpe

        dU_v = self.get_vib_energy_contribution(temperature)
        vprint(fmt % ('Cv_harm (0->T)', dU_v))
        U += dU_v

        vprint('-' * 31)
        vprint(fmt % ('U', U))
        vprint('=' * 31)
        return U


    def get_entropy(self, temperature, verbose=True):
        self.verbose = verbose
        vprint = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        vprint('Entropy components at T = %.2f K:' % temperature)
        vprint('=' * 49)
        vprint('%15s%13s     %13s' % ('', 'S', 'T*S'))
        S, S_dict = super().get_ideal_entropy(temperature=temperature, vibration=True)

        vprint(fmt % ('S_harm', S_dict['S_v'], S_dict['S_v'] * temperature))
        vprint('-' * 49)
        vprint(fmt % ('S', S, S * temperature))
        vprint('=' * 49)
        return S

    
    def get_helmholtz_energy(self, temperature, verbose=True)   ->float:
        """Calculate the Helmholtz free energy (eV) at a specified temperature (K)

        Inputs:
        temperature : float
            The temperature in Kelvin.
        verbose : bool
            If True, print the energy components.

        Returns:
        float : The Helmholtz free energy
        """
        #copy paste of HarmonicThermo.get_helmholtz_energy at the moment
        self.verbose = True
        vprint = self._vprint

        U = self.get_internal_energy(temperature, verbose=verbose)
        vprint('')
        S = self.get_entropy(temperature, verbose=verbose)
        F = U - temperature * S

        vprint('')
        vprint('Free energy components at T = %.2f K:' % temperature)
        vprint('=' * 23)
        fmt = '%5s%15.3f eV'
        vprint(fmt % ('U', U))
        vprint(fmt % ('-T*S', -temperature * S))
        vprint('-' * 23)
        vprint(fmt % ('F', F))
        vprint('=' * 23)
        return F


class MSRRHOThermo(QuasiHarmonicThermo):
    """Subclass of :class:`QuasiHarmonicThermo`, including Grimme's scaling method
    based on :doi:`10.1002/chem.201200497` and :doi:`10.1039/D1SC00621E`.

    Inputs:

    atoms: an ASE atoms object
        used to calculate rotational moments of inertia and molecular mass
    tau : float
        the vibrational energy threshold in :math:`cm^{-1}`, named
        :math:`\\tau` in :doi:`10.1039/D1SC00621E`.
        Values close or equal to 0 will result in the standard harmonic
        approximation. Defaults to :math:`35cm^{-1}`.
    nu_scal : float
        Linear scaling factor for the vibrational frequencies. Named
        :math:`\\nu_{scal}` in :doi:`10.1039/D1SC00621E`.
        Defaults to 1.0, check the `Truhlar group database
        <https://comp.chem.umn.edu/freqscale/index.html>`_
        for values corresponding to your level of theory.
        Note that for `\\nu_{scal}=1.0` this method is equivalent to
        the quasi-RRHO method in :doi:`10.1002/chem.201200497`.

    We enforce treating imaginary modes as Grimme suggests (converting 
    them to real by multiplying them with :math:`-i`).
    """

    def __init__(self, vib_energies, atoms, potentialenergy=0.,
                 tau=35, nu_scal=1.0) -> None:
        
        # Check for imaginary frequencies.
        vib_energies, n_imag = _clean_vib_energies(
            vib_energies, handling='invert'
        )
        # scale the frequencies (i.e. energies) before passing them on
        self.nu_scal = nu_scal
        self.vib_energies = np.multiply(vib_energies, self.nu_scal)
        # perhaps later we can pass the scaling to the class above
        self.n_imag = n_imag
        self.atoms = atoms
        self.tau = tau
        self.alpha = 4  # from paper 10.1002/chem.201200497
        # 1/s (tau to meters)
        self.tau_freq = units._c * self.tau * 1e2
        self.potentialenergy = potentialenergy


    def get_entropy(self, temperature, verbose=True) -> float:
        """Calculate the msRRHO entropy (eV/K) at a specified temperature (K)

        Inputs:
        temperature : float
            The temperature in Kelvin.
        verbose : bool
            If True, print the entropy components.

        Returns:
        float : The entropy in eV/K.
        """
        # overwrite verbosity to avoid double printing
        S = self.get_msRRHO_vib_entropy_v_contribution(temperature)
        vprint = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        vprint('Entropy components at T = %.2f K:' % temperature)
        vprint('=' * 49)
        vprint('%15s%13s     %13s' % ('', 'S', 'T*S'))
        vprint(fmt % ('S_vib', S, S * temperature))

        S_r_damp = self.get_msRRHO_vib_entropy_r_contribution(temperature)
        S += S_r_damp
        vprint(fmt % ('S_rot', S_r_damp, S_r_damp * temperature))

        vprint('-' * 49)
        vprint(fmt % ('S_tot', S, S * temperature))
        vprint('=' * 49)
        return S
    


    def get_msRRHO_energy(self, temperature, verbose=True) -> float:
        """Calculate the msRRHO energy (eV) at a specified temperature (K)

        Inputs:
        temperature : float
            The temperature in Kelvin.
        verbose : bool
            If True, print the energy components.
        Returns:
        float : The energy in eV.
        """
        
        from scipy.special import iv
        self.verbose = verbose
        vprint = self._vprint
        fmt = '%-15s%13.3f eV'
        vprint('Internal energy components at T = %.2f K:' % temperature)
        vprint('=' * 31)

        U = 0
        vprint(fmt % ('E_pot', self.potentialenergy))
        U = self.potentialenergy

        H_corr = 0

        #Only for printing
        zpe = self.get_ZPE_correction()
        vprint(fmt % ('E_ZPE', zpe))

        #Only for printing 
        H_vib = self.get_vib_energy_contribution(temperature)
        vprint(fmt % ('Cv_harm (0->T)', H_vib))
           

        H_trans = 2.5 * units._k * units._Nav * temperature 
        vprint(fmt % ('E_trans', H_RT))
        H_corr += H_trans

        H_rot = 1.5 * units._k * units._Nav * temperature 
        vprint(fmt % ('E_rot', H_RT))
        H_corr += H_rot

        H_v_damp = self.get_damped_vib_energy_contribution(temperature)
        vprint(fmt % ('E_vib_damp', H_v_damp))
        H_corr += H_v_damp

        H_r_damp = self.get_damped_free_rotor_energy_contribution(temperature)
        vprint(fmt % ('E_rot_damp', H_r_damp))
        H_corr += H_r_damp
      
        H_RT = units._k * units._Nav * temperature 
        vprint(fmt % ('Thermal energy', H_RT))
        H_corr += H_RT
        
        return H_corr
  




class HinderedThermo(ThermoChem):
    """Class for calculating thermodynamic properties in the hindered
    translator and hindered rotor model where all but three degrees of
    freedom are treated as harmonic vibrations, two are treated as
    hindered translations, and one is treated as a hindered rotation.

    Inputs:

    vib_energies : list
        a list of all the vibrational energies of the adsorbate (e.g., from
        ase.vibrations.Vibrations.get_energies). If atoms is not provided,
        then the number of energies must match the number of degrees of freedom
        of the adsorbate; i.e., 3*n, where n is the number of atoms. Note
        that this class does not check that the user has supplied the
        correct number of energies.
        Units of energies are eV.
    trans_barrier_energy : float
        the translational energy barrier in eV. This is the barrier for an
        adsorbate to diffuse on the surface.
    rot_barrier_energy : float
        the rotational energy barrier in eV. This is the barrier for an
        adsorbate to rotate about an axis perpendicular to the surface.
    sitedensity : float
        density of surface sites in cm^-2
    rotationalminima : integer
        the number of equivalent minima for an adsorbate's full rotation.
        For example, 6 for an adsorbate on an fcc(111) top site
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this class
        can be interpreted as the energy corrections)
    mass : float
        the mass of the adsorbate in amu (if mass is unspecified, then it will
        be calculated from the atoms class)
    inertia : float
        the reduced moment of inertia of the adsorbate in amu*Ang^-2
        (if inertia is unspecified, then it will be calculated from the
        atoms class)
    atoms : an ASE atoms object
        used to calculate rotational moments of inertia and molecular mass
    symmetrynumber : integer
        symmetry number of the adsorbate. This is the number of symmetric arms
        of the adsorbate and depends upon how it is bound to the surface.
        For example, propane bound through its end carbon has a symmetry
        number of 1 but propane bound through its middle carbon has a symmetry
        number of 2. (if symmetrynumber is unspecified, then the default is 1)
    imag_modes_handling : string
        If 'remove', any imaginary frequencies present after the 3N-3 cut will
        be removed in the calculation of the thermochemical properties.
        If 'error' (default), an error will be raised if imaginary frequencies
        are present after the 3N-3 cut.
        If 'invert', the imaginary frequencies after the 3N-3 cut will be
        multiplied by -i.
    """

    def __init__(self, vib_energies, trans_barrier_energy, rot_barrier_energy,
                 sitedensity, rotationalminima, potentialenergy=0.,
                 mass=None, inertia=None, atoms=None, symmetrynumber=1,
                 imag_modes_handling='error'):

        self.trans_barrier_energy = trans_barrier_energy * units._e
        self.rot_barrier_energy = rot_barrier_energy * units._e
        self.area = 1. / sitedensity / 100.0**2
        self.rotationalminima = rotationalminima
        self.potentialenergy = potentialenergy
        self.atoms = atoms
        self.symmetry = symmetrynumber

        # Sort the vibrations
        vib_energies = list(vib_energies)
        vib_energies.sort(key=np.abs)

        # Keep only the relevant vibrational energies (3N-3)
        if atoms:
            vib_energies = vib_energies[-(3 * len(atoms) - 3):]
        else:
            vib_energies = vib_energies[-(len(vib_energies) - 3):]

        # Check for imaginary frequencies.
        vib_energies, n_imag = _clean_vib_energies(
            vib_energies, handling=imag_modes_handling
        )
        self.vib_energies = vib_energies
        self.n_imag = n_imag

        if (mass or atoms) and (inertia or atoms):
            if mass:
                self.mass = mass * units._amu
            elif atoms:
                self.mass = np.sum(atoms.get_masses()) * units._amu
            if inertia:
                self.inertia = inertia * units._amu / units.m**2
            elif atoms:
                self.inertia = (atoms.get_moments_of_inertia()[2] *
                                units._amu / units.m**2)
        else:
            raise RuntimeError('Either mass and inertia of the '
                               'adsorbate must be specified or '
                               'atoms must be specified.')

        # Calculate hindered translational and rotational frequencies
        self.freq_t = np.sqrt(self.trans_barrier_energy / (2 * self.mass *
                                                           self.area))
        self.freq_r = 1. / (2 * np.pi) * np.sqrt(self.rotationalminima**2 *
                                                 self.rot_barrier_energy /
                                                 (2 * self.inertia))

    def get_internal_energy(self, temperature, verbose=True):
        """Returns the internal energy (including the zero point energy),
        in eV, in the hindered translator and hindered rotor model at a
        specified temperature (K)."""

        from scipy.special import iv

        self.verbose = verbose
        vprint = self._vprint
        fmt = '%-15s%13.3f eV'
        vprint('Internal energy components at T = %.2f K:' % temperature)
        vprint('=' * 31)

        U = 0.

        vprint(fmt % ('E_pot', self.potentialenergy))
        U += self.potentialenergy

        # Translational Energy
        T_t = units._k * temperature / (units._hplanck * self.freq_t)
        R_t = self.trans_barrier_energy / (units._hplanck * self.freq_t)
        dU_t = 2 * (-1. / 2 - 1. / T_t / (2 + 16 * R_t) + R_t / 2 / T_t -
                    R_t / 2 / T_t *
                    iv(1, R_t / 2 / T_t) / iv(0, R_t / 2 / T_t) +
                    1. / T_t / (np.exp(1. / T_t) - 1))
        dU_t *= units.kB * temperature
        vprint(fmt % ('E_trans', dU_t))
        U += dU_t

        # Rotational Energy
        T_r = units._k * temperature / (units._hplanck * self.freq_r)
        R_r = self.rot_barrier_energy / (units._hplanck * self.freq_r)
        dU_r = (-1. / 2 - 1. / T_r / (2 + 16 * R_r) + R_r / 2 / T_r -
                R_r / 2 / T_r *
                iv(1, R_r / 2 / T_r) / iv(0, R_r / 2 / T_r) +
                1. / T_r / (np.exp(1. / T_r) - 1))
        dU_r *= units.kB * temperature
        vprint(fmt % ('E_rot', dU_r))
        U += dU_r

        # Vibrational Energy
        dU_v = self.get_vib_energy_contribution(temperature)
        vprint(fmt % ('E_vib', dU_v))
        U += dU_v

        # Zero Point Energy
        dU_zpe = self.get_zero_point_energy()
        vprint(fmt % ('E_ZPE', dU_zpe))
        U += dU_zpe

        vprint('-' * 31)
        vprint(fmt % ('U', U))
        vprint('=' * 31)
        return U

    def get_zero_point_energy(self, verbose=True):
        """Returns the zero point energy, in eV, in the hindered
        translator and hindered rotor model"""

        zpe_t = 2 * (1. / 2 * self.freq_t * units._hplanck / units._e)
        zpe_r = 1. / 2 * self.freq_r * units._hplanck / units._e
        zpe_v = self.get_ZPE_correction()
        zpe = zpe_t + zpe_r + zpe_v
        return zpe

    def get_entropy(self, temperature, verbose=True):
        """Returns the entropy, in eV/K, in the hindered translator
        and hindered rotor model at a specified temperature (K)."""

        from scipy.special import iv

        self.verbose = verbose
        vrpint = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        vrpint('Entropy components at T = %.2f K:' % temperature)
        vrpint('=' * 49)
        vrpint('%15s%13s     %13s' % ('', 'S', 'T*S'))

        S = 0.

        # Translational Entropy
        T_t = units._k * temperature / (units._hplanck * self.freq_t)
        R_t = self.trans_barrier_energy / (units._hplanck * self.freq_t)
        S_t = 2 * (-1. / 2 + 1. / 2 * np.log(np.pi * R_t / T_t) -
                   R_t / 2 / T_t *
                   iv(1, R_t / 2 / T_t) / iv(0, R_t / 2 / T_t) +
                   np.log(iv(0, R_t / 2 / T_t)) +
                   1. / T_t / (np.exp(1. / T_t) - 1) -
                   np.log(1 - np.exp(-1. / T_t)))
        S_t *= units.kB
        vrpint(fmt % ('S_trans', S_t, S_t * temperature))
        S += S_t

        # Rotational Entropy
        T_r = units._k * temperature / (units._hplanck * self.freq_r)
        R_r = self.rot_barrier_energy / (units._hplanck * self.freq_r)
        S_r = (-1. / 2 + 1. / 2 * np.log(np.pi * R_r / T_r) -
               np.log(self.symmetry) -
               R_r / 2 / T_r * iv(1, R_r / 2 / T_r) / iv(0, R_r / 2 / T_r) +
               np.log(iv(0, R_r / 2 / T_r)) +
               1. / T_r / (np.exp(1. / T_r) - 1) -
               np.log(1 - np.exp(-1. / T_r)))
        S_r *= units.kB
        vrpint(fmt % ('S_rot', S_r, S_r * temperature))
        S += S_r

        # Vibrational Entropy
        S_v = self.get_vib_entropy_contribution(temperature)
        vrpint(fmt % ('S_vib', S_v, S_v * temperature))
        S += S_v

        # Concentration Related Entropy
        N_over_A = np.exp(1. / 3) * (10.0**5 /
                                     (units._k * temperature))**(2. / 3)
        S_c = 1 - np.log(N_over_A) - np.log(self.area)
        S_c *= units.kB
        vrpint(fmt % ('S_con', S_c, S_c * temperature))
        S += S_c

        vrpint('-' * 49)
        vrpint(fmt % ('S', S, S * temperature))
        vrpint('=' * 49)
        return S

    def get_helmholtz_energy(self, temperature, verbose=True):
        """Returns the Helmholtz free energy, in eV, in the hindered
        translator and hindered rotor model at a specified temperature
        (K)."""

        self.verbose = True
        vprint = self._vprint

        U = self.get_internal_energy(temperature, verbose=verbose)
        vprint('')
        S = self.get_entropy(temperature, verbose=verbose)
        F = U - temperature * S

        vprint('')
        vprint('Free energy components at T = %.2f K:' % temperature)
        vprint('=' * 23)
        fmt = '%5s%15.3f eV'
        vprint(fmt % ('U', U))
        vprint(fmt % ('-T*S', -temperature * S))
        vprint('-' * 23)
        vprint(fmt % ('F', F))
        vprint('=' * 23)
        return F


class IdealGasThermo(ThermoChem):
    """Class for calculating thermodynamic properties of a molecule
    based on statistical mechanical treatments in the ideal gas
    approximation.

    Inputs for enthalpy calculations:

    vib_energies : list
        a list of the vibrational energies of the molecule (e.g., from
        ase.vibrations.Vibrations.get_energies). The number of vibrations
        used is automatically calculated by the geometry and the number of
        atoms. If more are specified than are needed, then the lowest
        numbered vibrations are neglected. If either atoms or natoms is
        unspecified, then uses the entire list. Units are eV.
    geometry : 'monatomic', 'linear', or 'nonlinear'
        geometry of the molecule
    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)
    natoms : integer
        the number of atoms, used along with 'geometry' to determine how
        many vibrations to use. (Not needed if an atoms object is supplied
        in 'atoms' or if the user desires the entire list of vibrations
        to be used.)

    Extra inputs needed for entropy / free energy calculations:

    atoms : an ASE atoms object
        used to calculate rotational moments of inertia and molecular mass
    symmetrynumber : integer
        symmetry number of the molecule. See, for example, Table 10.1 and
        Appendix B of C. Cramer "Essentials of Computational Chemistry",
        2nd Ed.
    spin : float
        the total electronic spin. (0 for molecules in which all electrons
        are paired, 0.5 for a free radical with a single unpaired electron,
        1.0 for a triplet with two unpaired electrons, such as O_2.)
    imag_modes_handling : string
        If 'remove', any imaginary frequencies present after the 3N-5/3N-6 cut
        will be removed in the calculation of the thermochemical properties.
        If 'error' (default), an error will be raised if imaginary frequencies
        are present after the 3N-5/3N-6 cut.
        If 'invert', the imaginary frequencies after the 3N-5/3N-6 cut will be
        multiplied by -i.

    """

    def __init__(self, vib_energies, geometry, potentialenergy=0.,
                 atoms=None, symmetrynumber=None, spin=None, natoms=None,
                 imag_modes_handling='error'):
        self.potentialenergy = potentialenergy
        self.geometry = geometry
        self.atoms = atoms
        self.sigma = symmetrynumber
        self.spin = spin
        if natoms is None and atoms:
            natoms = len(atoms)
        self.natoms = natoms

        # Sort the vibrations
        vib_energies = list(vib_energies)
        vib_energies.sort(key=np.abs)

        # Cut the vibrations to those needed from the geometry.
        if natoms:
            if geometry == 'nonlinear':
                vib_energies = vib_energies[-(3 * natoms - 6):]
            elif geometry == 'linear':
                vib_energies = vib_energies[-(3 * natoms - 5):]
            elif geometry == 'monatomic':
                vib_energies = []
            else:
                raise ValueError(f"Unsupported geometry: {geometry}")

        # Check for imaginary frequencies.
        vib_energies, n_imag = _clean_vib_energies(
            vib_energies, handling=imag_modes_handling
        )
        self.vib_energies = vib_energies
        self.n_imag = n_imag

        self.referencepressure = 1.0e5  # Pa

    def get_enthalpy(self, temperature, verbose=True):
        """Returns the enthalpy, in eV, in the ideal gas approximation
        at a specified temperature (K)."""

        self.verbose = verbose
        vprint = self._vprint
        fmt = '%-15s%13.3f eV'
        vprint('Enthalpy components at T = %.2f K:' % temperature)
        vprint('=' * 31)

        H = 0.

        vprint(fmt % ('E_pot', self.potentialenergy))
        H += self.potentialenergy

        zpe = self.get_ZPE_correction()
        vprint(fmt % ('E_ZPE', zpe))
        H += zpe

        Cv_tT = self.get_ideal_translational_energy(temperature)
        vprint(fmt % ('Cv_trans (0->T)', Cv_tT))
        H += Cv_tT

        Cv_rT = self.get_ideal_rotational_energy(self.geometry, temperature)
        vprint(fmt % ('Cv_rot (0->T)', Cv_rT ))
        H += Cv_rT

        dH_v = self.get_vib_energy_contribution(temperature)
        vprint(fmt % ('Cv_vib (0->T)', dH_v))
        H += dH_v

        Cp_corr = units.kB * temperature
        vprint(fmt % ('(C_v -> C_p)', Cp_corr))
        H += Cp_corr

        vprint('-' * 31)
        vprint(fmt % ('H', H))
        vprint('=' * 31)
        return H

    def get_entropy(self, temperature, pressure, verbose=True):
        """Returns the entropy, in eV/K, in the ideal gas approximation
        at a specified temperature (K) and pressure (Pa)."""

        if self.atoms is None or self.sigma is None or self.spin is None:
            raise RuntimeError('atoms, symmetrynumber, and spin must be '
                               'specified for entropy and free energy '
                               'calculations.')
        self.verbose = verbose
        vprint = self._vprint
        fmt = '%-15s%13.7f eV/K%13.3f eV'
        vprint('Entropy components at T = %.2f K and P = %.1f Pa:' %
              (temperature, pressure))
        vprint('=' * 49)
        vprint('%15s%13s     %13s' % ('', 'S', 'T*S'))

        S, S_dict = super().get_ideal_entropy(temperature,
                                translation=True, vibration=True,
                                rotation=True, geometry=self.geometry,
                                electronic=True,
                                pressure=pressure)

        vprint(fmt % ('S_trans (1 bar)', S_dict['S_t'], S_dict['S_t'] * temperature))
        vprint(fmt % ('S_rot', S_dict['S_r'], S_dict['S_r'] * temperature))
        vprint(fmt % ('S_elec', S_dict['S_e'], S_dict['S_e'] * temperature))
        vprint(fmt % ('S_vib', S_dict['S_v'], S_dict['S_v'] * temperature))
        vprint(fmt % ('S (1 bar -> P)', S_dict['S_p'], S_dict['S_p'] * temperature))
        vprint('-' * 49)
        vprint(fmt % ('S', S, S * temperature))
        vprint('=' * 49)
        return S

    def get_gibbs_energy(self, temperature, pressure, verbose=True):
        """Returns the Gibbs free energy, in eV, in the ideal gas
        approximation at a specified temperature (K) and pressure (Pa)."""

        self.verbose = verbose
        vprint = self._vprint

        H = self.get_enthalpy(temperature, verbose=verbose)
        vprint('')
        S = self.get_entropy(temperature, pressure, verbose=verbose)
        G = H - temperature * S

        vprint('')
        vprint('Free energy components at T = %.2f K and P = %.1f Pa:' %
              (temperature, pressure))
        vprint('=' * 23)
        fmt = '%5s%15.3f eV'
        vprint(fmt % ('H', H))
        vprint(fmt % ('-T*S', -temperature * S))
        vprint('-' * 23)
        vprint(fmt % ('G', G))
        vprint('=' * 23)
        return G


class CrystalThermo(ThermoChem):
    """Class for calculating thermodynamic properties of a crystalline
    solid in the approximation that a lattice of N atoms behaves as a
    system of 3N independent harmonic oscillators.

    Inputs:

    phonon_DOS : list
        a list of the phonon density of states,
        where each value represents the phonon DOS at the vibrational energy
        value of the corresponding index in phonon_energies.

    phonon_energies : list
        a list of the range of vibrational energies (hbar*omega) over which
        the phonon density of states has been evaluated. This list should be
        the same length as phonon_DOS and integrating phonon_DOS over
        phonon_energies should yield approximately 3N, where N is the number
        of atoms per unit cell. If the first element of this list is
        zero-valued it will be deleted along with the first element of
        phonon_DOS. Units of vibrational energies are eV.

    potentialenergy : float
        the potential energy in eV (e.g., from atoms.get_potential_energy)
        (if potentialenergy is unspecified, then the methods of this
        class can be interpreted as the energy corrections)

    formula_units : int
        the number of formula units per unit cell. If unspecified, the
        thermodynamic quantities calculated will be listed on a
        per-unit-cell basis.
    """

    def __init__(self, phonon_DOS, phonon_energies,
                 formula_units=None, potentialenergy=0.):
        self.phonon_energies = phonon_energies
        self.phonon_DOS = phonon_DOS

        if formula_units:
            self.formula_units = formula_units
            self.potentialenergy = potentialenergy / formula_units
        else:
            self.formula_units = 0
            self.potentialenergy = potentialenergy

    def get_internal_energy(self, temperature, verbose=True):
        """Returns the internal energy, in eV, of crystalline solid
        at a specified temperature (K)."""

        self.verbose = verbose
        vprint = self._vprint
        fmt = '%-15s%13.4f eV'
        if self.formula_units == 0:
            vprint('Internal energy components at '
                  'T = %.2f K,\non a per-unit-cell basis:' % temperature)
        else:
            vprint('Internal energy components at '
                  'T = %.2f K,\non a per-formula-unit basis:' % temperature)
        vprint('=' * 31)

        U = 0.

        omega_e = self.phonon_energies
        dos_e = self.phonon_DOS
        if omega_e[0] == 0.:
            omega_e = np.delete(omega_e, 0)
            dos_e = np.delete(dos_e, 0)

        vprint(fmt % ('E_pot', self.potentialenergy))
        U += self.potentialenergy

        zpe_list = omega_e / 2.
        if self.formula_units == 0:
            zpe = np.trapz(zpe_list * dos_e, omega_e)
        else:
            zpe = np.trapz(zpe_list * dos_e, omega_e) / self.formula_units
        vprint(fmt % ('E_ZPE', zpe))
        U += zpe

        B = 1. / (units.kB * temperature)
        E_vib = omega_e / (np.exp(omega_e * B) - 1.)
        if self.formula_units == 0:
            E_phonon = np.trapz(E_vib * dos_e, omega_e)
        else:
            E_phonon = np.trapz(E_vib * dos_e, omega_e) / self.formula_units
        vprint(fmt % ('E_phonon', E_phonon))
        U += E_phonon

        vprint('-' * 31)
        vprint(fmt % ('U', U))
        vprint('=' * 31)
        return U

    def get_entropy(self, temperature, verbose=True):
        """Returns the entropy, in eV/K, of crystalline solid
        at a specified temperature (K)."""

        self.verbose = verbose
        vprint = self._vprint
        fmt = '%-15s%13.7f eV/K%13.4f eV'
        if self.formula_units == 0:
            vprint('Entropy components at '
                  'T = %.2f K,\non a per-unit-cell basis:' % temperature)
        else:
            vprint('Entropy components at '
                  'T = %.2f K,\non a per-formula-unit basis:' % temperature)
        vprint('=' * 49)
        vprint('%15s%13s     %13s' % ('', 'S', 'T*S'))

        omega_e = self.phonon_energies
        dos_e = self.phonon_DOS
        if omega_e[0] == 0.:
            omega_e = np.delete(omega_e, 0)
            dos_e = np.delete(dos_e, 0)

        B = 1. / (units.kB * temperature)
        S_vib = (omega_e / (temperature * (np.exp(omega_e * B) - 1.)) -
                 units.kB * np.log(1. - np.exp(-omega_e * B)))
        if self.formula_units == 0:
            S = np.trapz(S_vib * dos_e, omega_e)
        else:
            S = np.trapz(S_vib * dos_e, omega_e) / self.formula_units

        vprint('-' * 49)
        vprint(fmt % ('S', S, S * temperature))
        vprint('=' * 49)
        return S

    def get_helmholtz_energy(self, temperature, verbose=True):
        """Returns the Helmholtz free energy, in eV, of crystalline solid
        at a specified temperature (K)."""

        self.verbose = True
        vprint = self._vprint

        U = self.get_internal_energy(temperature, verbose=verbose)
        vprint('')
        S = self.get_entropy(temperature, verbose=verbose)
        F = U - temperature * S

        vprint('')
        if self.formula_units == 0:
            vprint('Helmholtz free energy components at '
                  'T = %.2f K,\non a per-unit-cell basis:' % temperature)
        else:
            vprint('Helmholtz free energy components at '
                  'T = %.2f K,\non a per-formula-unit basis:' % temperature)
        vprint('=' * 23)
        fmt = '%5s%15.4f eV'
        vprint(fmt % ('U', U))
        vprint(fmt % ('-T*S', -temperature * S))
        vprint('-' * 23)
        vprint(fmt % ('F', F))
        vprint('=' * 23)
        return F


def _clean_vib_energies(vib_energies, handling='error',
                        value=None) -> Tuple[List[float], int]:
    """Checks and deal with the presence of imaginary vibrational modes

    Also removes +0.j from real vibrational energies.

    Inputs:

    vib_energies : list
        a list of the vibrational energies

    handling : string
        If 'remove', any imaginary frequencies will be removed.
        If 'error' (default), an error will be raised if imaginary
        frequencies are present.
        If 'invert', the imaginary part of the frequencies will be
        multiplied by -i. See :doi:`10.1002/anie.202205735.`
        If 'raise', all imaginary frequencies will be replaced with the value
        specified by the 'value' argument. See Cramer, Truhlar and coworkers.
        specified by the 'value' argument. See Cramer, Truhlar and coworkers.
        :doi:`10.1021/jp205508z`.

    Outputs:

    vib_energies : list
        a list of the real vibrational energies.

    n_imag : int
        the number of imaginary frequencies treated.
    """
    if handling.lower() == 'remove':
        n_vib_energies = len(vib_energies)
        vib_energies = [v for v in vib_energies if np.real(v) > 0]
        n_imag = n_vib_energies - len(vib_energies)
        if n_imag > 0:
            warn(f"{n_imag} imag modes removed", UserWarning)
    elif handling.lower() == 'error':
        if sum(np.iscomplex(vib_energies)):
            raise ValueError('Imaginary vibrational energies are present.')
        n_imag = 0
    elif handling.lower() == 'invert':
        n_imag = sum(np.iscomplex(vib_energies))
        vib_energies = [np.imag(v) if np.iscomplex(v)
                        else v for v in vib_energies]
    elif handling.lower() == 'raise':
        if value is None:
            raise ValueError("Value must be specified when handling='raise'.")
        n_imag = sum(np.iscomplex(vib_energies))
        vib_energies = [value if np.iscomplex(v)
                        else v for v in vib_energies]
    else:
        raise ValueError(f"Unknown handling option: {handling}")
    vib_energies = np.real(vib_energies).tolist()  # clear +0.j

    return vib_energies, n_imag
