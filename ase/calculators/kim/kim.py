"""
Knowledgebase of Interatomic Models (KIM) Calculator for ASE written by:

Ellad B. Tadmor
Mingjian Wen
University of Minnesota

This calculator selects an appropriate calculator for a KIM model depending on
whether it supports the KIM application programming interface (API) or is a
KIM Simulator Model. For more information on KIM, visit https://openkim.org.
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import re
import os
from ase.data import atomic_masses, atomic_numbers
try:
    from kimpy import simulator_models as kimsm
    kimsm_loaded = True
except Exception:
    kimsm_loaded = False
from ase.calculators.lammpslib import LAMMPSlib
from ase.calculators.lammpsrun import LAMMPS
from .kimmodel import KIMModelCalculator
from .exceptions import KIMCalculatorError


def KIM(extended_kim_id, simulator=None, options=None, debug=False):
    """
    Calculator function for interatomic models archived in the Open Knowledgebase
    of Interatomic Models (OpenKIM) at https://openkim.org

    Parameters
    ----------

    extended_kim_id: string
      Extended KIM ID of the KIM interatomic model (for details see:
      https://openkim.org/about-kim-ids/)

    simulator: string
      Name of the simulator to be used. This is the name of the ASE calculator
      that will be used to run the KIM interatomic model.

      If ``None``, the simulator is determined automatically by the function.
      Supported simulators are: ``kimmodel``, ``lammpsrun``, and ``lammpslib``.
      (Note that since LAMMPS is compatible with the KIM API, LAMMPS calculators
      can also be used to run KIM models.)

    options: dictionary
      Additional options passed to the initializer of the selected simulator.

      If the simulator is ``kimmodel``, possible options are:

      options = {'neigh_skin_ratio': 0.2, 'release_GIL': False}

      where ``neigh_skin_ratio`` provides the skin (in percentage of cutoff) used to
      determine the neighbor list, and ``release_GIL`` determines whether to
      release the python GIL, which allows a KIM model to be run with multiple threads.

      See the LAMMPS calculators doc page
      https://wiki.fysik.dtu.dk/ase/ase/calculators/lammps.html
      for available options for ``lammpsrun`` and ``lammpslib``.

    debug: bool
      If ``True``, turn on the debug mode to output extra information.


    This function returns a calculator based on the argument values of
    ``extended_kim_id`` and ``simulator``.

    """

    # options set internally in this calculator
    kimmodel_not_allowed_options = ['modelname', 'debug']
    lammpsrun_not_allowed_options = ['parameters', 'files', 'specorder',
                                     'keep_tmp_files', 'has_charges']
    lammpslib_not_allowed_options = ['lammps_header', 'lmpcmds',
                                     'atom_types', 'log_file', 'keep_alive']
    asap_kimmo_not_allowed_options = ['name', 'verbose']
    asap_simmo_not_allowed_options = ['Params']
    if options is None:
        options = dict()

    # Determine whether this is a standard KIM Model or a KIM Simulator Model
    kim_id, this_is_a_KIM_MO = _get_kim_model_id_and_type(extended_kim_id)

    # If this is a KIM Model (supports KIM API) return support through
    # a KIM-compliant simulator
    if this_is_a_KIM_MO:

        if simulator is None:   # Default
            simulator = 'kimmodel'
        else:
            simulator = simulator.lower().strip()

        if simulator == 'kimmodel':
            msg = _check_conflict_options(
                options, kimmodel_not_allowed_options, simulator)
            if msg is not None:
                raise KIMCalculatorError(msg)
            else:
                return KIMModelCalculator(extended_kim_id, debug=debug)

        elif simulator == 'asap':
            try:
                from asap3 import OpenKIMcalculator
            except ImportError as e:
                raise ImportError(str(e) + ' You need to install asap3 first.')

            msg = _check_conflict_options(
                options, asap_kimmo_not_allowed_options, simulator)
            if msg is not None:
                raise KIMCalculatorError(msg)
            else:
                return(OpenKIMcalculator(name=extended_kim_id, verbose=debug))

        elif simulator == 'lammpsrun':
            # check options
            msg = _check_conflict_options(
                options, lammpsrun_not_allowed_options, simulator)
            if msg is not None:
                raise KIMCalculatorError(msg)

            supported_species = KIM_get_supported_species_list(extended_kim_id)
            param_filenames = []  # no parameter files to pass
            parameters = {}
            parameters['pair_style'] = 'kim ' + \
                extended_kim_id.strip() + os.linesep
            parameters['pair_coeff'] = [
                '* * ' + ' '.join(supported_species) + os.linesep]
            parameters['model_init'] = []
            parameters['model_post'] = []
            parameters['mass'] = []
            for i, species in enumerate(supported_species):
                if species not in atomic_numbers:
                    raise KIMCalculatorError(
                        'Unknown element species {0}.'.format(species))
                massstr = str(atomic_masses[atomic_numbers[species]])
                parameters['mass'].append(str(i + 1) + " " + massstr)

            # Return LAMMPS calculator
            return LAMMPS(parameters=parameters, files=param_filenames,
                          specorder=supported_species, keep_tmp_files=debug)

        # TODO add lammps_lib
        elif simulator == 'lammpslib':
            raise KIMCalculatorError(
                'lammpslib does not support KIM model. try "lammpsrun".')
        else:
            raise KIMCalculatorError(
                'ERROR: Unsupported simulator "%s" requested to run KIM API '
                'compliant KIM Models.' % simulator)

    # If we get to here, the model is a KIM Simulator Model ###

    # Initialize KIM SM object
    ksm = kimsm.ksm_object(extended_kim_id=extended_kim_id)
    param_filenames = ksm.get_model_param_filenames()

    # Double check that the extended KIM ID of the Simulator Model
    # matches the expected value. (If not, the KIM SM is corrupted.)
    SM_extended_kim_id = ksm.get_model_extended_kim_id()
    if extended_kim_id != SM_extended_kim_id:
        raise KIMCalculatorError(
            'ERROR: SM extended KIM ID ("%s") does not match expected value '
            ' ("%s").' % (SM_extended_kim_id, extended_kim_id))

    # Get simulator name
    simulator_name = ksm.get_model_simulator_name().lower()

    # determine simulator
    if simulator is None:
        if simulator_name == 'asap':
            simulator = 'asap'
        elif simulator_name == 'lammps':
            simulator = 'lammpslib'

    #  Get model definition from SM metadata
    model_defn = ksm.get_model_defn_lines()
    if len(model_defn) == 0:
        raise KIMCalculatorError(
            'ERROR: model-defn is an empty list in metadata file of '
            'Simulator Model "%s".' % extended_kim_id)
    if "" in model_defn:
        raise KIMCalculatorError(
            'ERROR: model-defn contains one or more empty strings in metadata '
            'file of Simulator Model "%s".' % extended_kim_id)

    if simulator_name == "asap":
        try:
            from asap3 import EMT, EMTMetalGlassParameters, EMTRasmussenParameters
        except ImportError as e:
            raise ImportError(str(e) + ' You need to install asap3 first.')

        # check options
        msg = _check_conflict_options(
            options, asap_simmo_not_allowed_options, simulator)
        if msg is not None:
            raise KIMCalculatorError(msg)

        # Verify units (ASAP models are expected to work with "ase" units)
        supported_units = ksm.get_model_units().lower().strip()
        if supported_units != "ase":
            raise KIMCalculatorError(
                'ERROR: KIM Simulator Model units are "%s", but expected to '
                'be "ase" for ASAP.' % supported_units)

        # There should be only one model_defn line
        if len(model_defn) != 1:
            raise KIMCalculatorError(
                'ERROR: model-defn contains %d lines, but should only contain '
                'one line for an ASAP model.' % len(model_defn))

        # Return calculator
        unknown_potential = False
        if model_defn[0].lower().strip().startswith("emt"):
            # pull out potential parameters
            pp = ''
            mobj = re.search(r"\(([A-Za-z0-9_\(\)]+)\)", model_defn[0])
            if mobj is not None:
                pp = mobj.group(1).strip().lower()
            if pp == '':
                calc = EMT()
            elif pp.startswith('emtrasmussenparameters'):
                calc = EMT(Params=EMTRasmussenParameters())
            elif pp.startswith('emtmetalglassparameters'):
                calc = EMT(Params=EMTMetalGlassParameters())
            else:
                unknown_potential = True

        if unknown_potential:
            raise KIMCalculatorError(
                'ERROR: Unknown model "%s" for simulator ASAP.' % model_defn[0])
        else:
            calc.set_subtractE0(False)  # Use undocumented feature for the EMT
            # calculators to take the energy of an
            # isolated atoms as zero. (Otherwise it
            # is taken to be that of perfect FCC.)
            return calc

    elif simulator_name == "lammps":

        param_filenames_for_lammps = list(param_filenames)
        if simulator == 'lammpsrun':
            # Remove path from parameter file names since lammpsrun copies all
            # files into a tmp directory, so path should not appear on
            # in LAMMPS commands
            param_filenames_for_lammps = [os.path.basename(i)
                                          for i in param_filenames_for_lammps]

        # Build atom species and type lists based on all supported species.
        # This means that the LAMMPS simulation will be defined to have
        # as many atom types as are supported by the SM and each atom will
        # be assigned a type based on its species (in the order that the
        # species are defined in the SM).
        supported_species = ksm.get_model_supported_species()
        atom_type_sym_list_string = ' '.join(supported_species)
        atom_type_num_list_string = ' '.join(
            [str(atomic_numbers[s]) for s in supported_species])

        # Process KIM templates in model_defn lines
        for i in range(0, len(model_defn)):
            model_defn[i] = kimsm.template_substitution(
                model_defn[i], param_filenames_for_lammps, ksm.sm_dirname,
                atom_type_sym_list_string, atom_type_num_list_string)

        # Get model init lines
        model_init = ksm.get_model_init_lines()

        # Process KIM templates in model_init lines
        for i in range(0, len(model_init)):
            model_init[i] = kimsm.template_substitution(
                model_init[i], param_filenames_for_lammps, ksm.sm_dirname,
                atom_type_sym_list_string, atom_type_num_list_string)

        # Get model supported units
        supported_units = ksm.get_model_units().lower().strip()

        if simulator == 'lammpsrun':
            # check options
            msg = _check_conflict_options(
                options, lammpsrun_not_allowed_options, simulator)
            if msg is not None:
                raise KIMCalculatorError(msg)

            # add cross-platform line separation to model definition lines
            model_defn = [s + os.linesep for s in model_defn]

            # Extract parameters for LAMMPS calculator from model definition lines
            parameters = _get_params_for_LAMMPS_calculator(model_defn,
                                                           supported_species)

            # Add units to parameters
            parameters["units"] = supported_units

            # add cross-platform line separation to model definition lines
            model_init = [s + os.linesep for s in model_init]

            # Add init lines to parameter list
            _add_init_lines_to_parameters(parameters, model_init)

            # Determine whether this model has charges
            has_charges = False
            for ii, mline in enumerate(model_init):
                ml = re.sub(' +', ' ', mline).strip().lower()
                if ml.startswith('atom_style charge'):
                    has_charges = True

            # Return LAMMPS calculator
            return LAMMPS(parameters=parameters, files=param_filenames,
                          specorder=supported_species, keep_tmp_files=debug,
                          has_charges=has_charges)

        elif simulator == 'lammpslib':
            # check options
            msg = _check_conflict_options(
                options, lammpslib_not_allowed_options, simulator)
            if msg is not None:
                raise KIMCalculatorError(msg)

            # Setup LAMMPS header commands lookup table
            model_init.insert(0, 'atom_modify map array sort 0 0')
            if not any("atom_style" in s.lower() for s in model_init):
                model_init.insert(0, 'atom_style atomic')
            model_init.insert(
                0, 'units ' + supported_units.strip())     # units

            # Assign atom types to species
            atom_types = {}
            for i_s, s in enumerate(supported_species):
                atom_types[s] = i_s + 1

            # Return LAMMPSlib calculator
            return LAMMPSlib(lammps_header=model_init,
                             lammps_name=None,
                             lmpcmds=model_defn,
                             atom_types=atom_types,
                             log_file='lammps.log',
                             keep_alive=True)

        else:
            raise KIMCalculatorError(
                'ERROR: Unknown LAMMPS calculator: "%s".' % simulator)

    else:
        raise KIMCalculatorError(
            'ERROR: Unsupported simulator: "%s".' % simulator_name)


def _get_kim_model_id_and_type(extended_kim_id):
    '''
    Determine whether "extended_kim_id" corresponds to either a KIM Model
    or KIM Simulator Model and extract the short KIM ID
    '''
    # Determine whether this is a KIM Model or SM
    if kimsm.is_simulator_model(extended_kim_id):
        if not kimsm_loaded:
            raise KIMCalculatorError('ERROR: Model % s is a Simulator Model, '
                                     'but "kimpy.simulator_models" is not loaded.' % extended_kim_id)
        this_is_a_KIM_MO = False
        pref = 'SM'
    else:
        this_is_a_KIM_MO = True
        pref = 'MO'
    # Try to parse model name assuming it has an extended KIM ID format
    # to obtain short KIM_ID. This is used to name the directory
    # containing the SM files.
    extended_kim_id_regex = pref + '_[0-9]{12}_[0-9]{3}'
    try:
        kim_id = re.search(extended_kim_id_regex, extended_kim_id).group(0)
    except AttributeError:
        kim_id = extended_kim_id  # Model name does not contain a short KIM ID,
        # so use full model name for the file directory.

    return kim_id, this_is_a_KIM_MO


def _get_params_for_LAMMPS_calculator(model_defn, supported_species):
    '''
    Extract parameters for LAMMPS calculator from model definition lines.
    Returns a dictionary with entries for "pair_style" and "pair_coeff".
    Expects there to be only one "pair_style" line. There can be multiple
    "pair_coeff" lines (result is returned as a list).
    '''
    parameters = {}
    parameters['pair_style'] = ''
    parameters['pair_coeff'] = []
    parameters['model_post'] = []
    found_pair_style = False
    found_pair_coeff = False
    for i in range(0, len(model_defn)):
        c = model_defn[i]
        if c.lower().startswith('pair_style'):
            if found_pair_style:
                raise KIMCalculatorError(
                    'ERROR: More than one pair_style in metadata file.')
            found_pair_style = True
            parameters['pair_style'] = c.split(" ", 1)[1]
        elif c.lower().startswith('pair_coeff'):
            found_pair_coeff = True
            parameters['pair_coeff'].append(c.split(" ", 1)[1])
        else:
            parameters['model_post'].append(c)
    if not found_pair_style:
        raise KIMCalculatorError(
            'ERROR: pair_style not found in metadata file.')
    if not found_pair_coeff:
        raise KIMCalculatorError(
            'ERROR: pair_coeff not found in metadata file.')

    #  For every species in "supported_species", add an entry to the "mass" key in
    #  dictionary "parameters".
    parameters['mass'] = []
    for i, species in enumerate(supported_species):
        if species not in atomic_numbers:
            raise KIMCalculatorError(
                'Unknown element species {0}.'.format(species))
        massstr = str(atomic_masses[atomic_numbers[species]])
        parameters['mass'].append(str(i + 1) + " " + massstr)

    return parameters


def _add_init_lines_to_parameters(parameters, model_init):
    '''
    Add Simulator Model initialization lines to the parameter list for LAMMPS
    if there are any.
    '''
    parameters['model_init'] = []
    for i in range(0, len(model_init)):
        parameters['model_init'].append(model_init[i])


def _check_conflict_options(options, not_allowed_options, simulator):
    """Check whether options is in not_allowed options"""
    s1 = set(options)
    s2 = set(not_allowed_options)
    common = s1.intersection(s2)
    if common:
        msg1 = 'Simulator "{}" does not support argument(s): '.format(
            simulator)
        msg2 = ', '.join(['"{}"'.format(s) for s in common])
        msg3 = ' provided in "options", because it is (they are) determined '
        msg4 = 'internally within the KIM calculator.'
        return msg1 + msg2 + msg3 + msg4
    else:
        msg = None
    return msg


def KIM_get_supported_species_list(extended_kim_id, simulator='kimmodel'):
    '''
    Returns a list of the atomic species (element names) supported by the
    specified KIM Model or KIM Supported Model.

    extended_kim_id: string
       Extended KIM ID of the model to be calculated

    simulator: string
       Name of simulator to be used for obtaining the list of model species
       Available options: kimmodel (default), asap
    '''
    # Determine whether this is a standard KIM Model or
    # a KIM Simulator Model
    kim_id, this_is_a_KIM_MO = _get_kim_model_id_and_type(extended_kim_id)

    # If this is a KIM Model, get supported species list
    if this_is_a_KIM_MO:

        if simulator == 'kimmodel':

            calc = KIMModelCalculator(extended_kim_id)
            speclist = list(calc.get_kim_model_supported_species())

        elif simulator == 'asap':
            try:
                from asap3 import OpenKIMcalculator
            except ImportError as e:
                raise ImportError(str(e) + ' You need to install asap3 first.')
            calc = OpenKIMcalculator(extended_kim_id)
            speclist = list(calc.get_kim_model_supported_species())

        else:
            raise KIMCalculatorError(
                'ERROR: Unsupported simulator "%s" requested to obtain KIM '
                'Model species list.' % simulator)

    # Otherwise this is an SM and we'll get the supported species list from metadata
    else:

        # Initialize KIM SM object
        ksm = kimsm.ksm_object(extended_kim_id=extended_kim_id)
        speclist = ksm.get_model_supported_species()

    # Return list of supported species
    return speclist
