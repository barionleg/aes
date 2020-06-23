""" This module defines io functions for MuST calculator"""

from ase import Atoms
from ase.units import Bohr, Rydberg
from ase.io.must.default_params import defaults
from ase.data import atomic_numbers

magmoms = {'Fe': 2.1,
           'Co': 1.4,
           'Ni': 0.6}


def write_positions_input(atoms, method):
    """ Function that writes the positions data input file based on
    atoms object and selected calculation method"""
    with open('position.dat', 'w') as filehandle:
        filehandle.write(str(1.0) + '\n\n')

        for i in range(3):
            filehandle.write('%s\n' % str(atoms.get_cell()[i] / Bohr)[1:-1])
        filehandle.write('\n')

        if method == 3:

            for site in atoms.info['CPA']:
                sitestring = 'CPA  %s' % str(atoms[site['index']].position
                                             / Bohr)[1:-1]

                for key in site.keys():
                    if key == 'index':
                        pass
                    else:
                        sitestring += '  %s %s' % (key, str(site[key]))
                sitestring += '\n'
                filehandle.write(sitestring)

        else:
            for index in range(len(atoms)):
                filehandle.write('%s %s\n'
                                 % (atoms[index].symbol,
                                    str(atoms[index].position / Bohr)[1:-1]))


def write_atomic_pot_input(symbol, nspins, moment, xc, niter, mp):
    """
    Function to write input file for generating
    atomic potential using 'newa' command
    Parameters
    ----------
    symbol: str
        Chemical symbol of the element.
    nspins: int
        Number of spins.
    moment: float
        Magnetic moment.
    xc: int
        ex-cor type (1=vb-hedin,2=vosko).
    niter: int
        Maximum number of SCF iterations.
    mp: float
        SCF mixing parameter
    """
    title = symbol + ' Atomic Potential'
    output_file = symbol + '_a_out'
    pot_file = symbol + '_a_pot'
    z = atomic_numbers[symbol]

    if moment == 0. and nspins == 2 and symbol in ['Fe', 'Co', 'Ni']:
        moment = magmoms[symbol]

    space = '                    '
    contents = ['Title:' + title,
                output_file + space +
                'Output file name. If blank, data will show on screen',
                str(z) + space + 'Atomic number', str(moment) + space
                + 'Magnetic moment',
                str(nspins) + space + 'Number of spins',
                str(xc) + space +
                'Exchange-correlation type (1=vb-hedin,2=vosko)',
                str(niter) + space + 'Number of Iterations',
                str(mp) + space + 'Mixing parameter',
                str(pot_file) + space + 'Output potential file']

    with open(symbol + '_a_in', 'w') as filehandle:
        for entry in contents:
            filehandle.write('%s\n' % entry)


def write_single_site_pot_input(symbol, crystal_type, a, nspins, moment, xc,
                                lmax, print_level, ncomp, conc, mt_radius,
                                ws_radius, egrid, ef, niter, mp):
    """
    Function to write input file for generating single site
    potential using 'newss' command
    Parameters
    ----------

    symbol: str
        Chemical symbol of the element
    crystal_type: int
                1 for FCC, 2 for BCC.
    a: float
        The lattice constant.
    nspins: int
        number of spins.
    moment: float
        Magnetic moment. If nspins = 2 and moment = 0 during input,
        moment will be changed to values from
        this dictionary: {Fe': 2.1, 'Co': 1.4, 'Ni': 0.6}
    xc: int
        ex-cor type (1=vb-hedin,2=vosko).
    lmax: int
        angular momentum quantum number cutoff value.
    print_level: int
        Print level.
    ncomp: int
        Number of components.
    conc: float
        Concentrations.
    mt_radius: float
            mt_radius.
    ws_radius: float
        ws_radius.
    egrid: vector
        e-grid vector of form (ndiv(=#div/0.1Ryd), bott, eimag).
    ef: float
        Estomate of fermi energy.
    niter: int
        Maximum number of SCF iterations.
    mp: float
        Mixing parameter for SCF iterations.
    """

    title = symbol + ' Single Site Potential'
    output_file = symbol + '_ss_out'
    input_file = symbol + '_a_pot'
    pot_file = symbol + '_ss_pot'
    keep_file = symbol + '_ss_k'

    z = atomic_numbers[symbol]
    a = a / Bohr

    if moment == 0.:
        if nspins == 2:
            if symbol in ['Fe', 'Co', 'Ni']:
                moment = magmoms[symbol]

    space = '                    '
    contents = [title, output_file + space +
                'Output file name. If blank, data will show on screen',
                str(print_level) + space + 'Print level', str(crystal_type) +
                space + 'Crystal type (1=FCC,2=BCC)',
                str(lmax) + space + 'lmax', str(a) + space +
                'Lattice constant',
                str(nspins) + space + 'Number of spins',
                str(xc) + space +
                'Exchange Correlation type (1=vb-hedin,2=vosko)',
                str(ncomp) + space + 'Number of components', str(z) + '  ' +
                str(moment) + space + 'Atomic number, Magnetic moment',
                str(conc) + space + 'Concentrations',
                str(mt_radius / Bohr) + '  ' + str(ws_radius / Bohr) +
                space + 'mt radius, ws radius',
                str(input_file) + space + 'Input potential file',
                str(pot_file) + space + 'Output potential file',
                str(keep_file) + space + 'Keep file',
                str(egrid[0]) + ' ' + str(egrid[1]) + ' ' + str(egrid[2]) +
                space + 'e-grid: ndiv(=#div/0.1Ryd), bott, eimag',
                str(ef / Rydberg) + ' ' + str(ef / Rydberg) +
                space + 'Fermi energy (estimate)',
                str(niter) + ' ' + str(mp) + space +
                'Number of scf iterations, Mixing parameter']

    with open(str(symbol) + '_ss_in', 'w') as filehandle:
        for entry in contents:
            filehandle.write('%s\n' % entry)


def write_input_parameters_file(atoms, parameters):
    """Write the main input file for 'mst2' command. This file contains all
    essential input parameters required for calculation"""
    energy_params = ['etol', 'ptol', 'ftol',
                     'offset_energy_pt',
                     'em_switch']  # Parameters with units of energy
    spatial_params = ['liz_cutoff', 'max_core_radius',
                      'max_mt_radius', 'core_radius',
                      'mt_radius']  # Parameters with units of length
    vector_params = ['uniform_grid', 'grid_origin', 'grid_1',
                     'grid_2', 'grid_3', 'grid_pts', 'kpts',
                     'moment_direction', 'constrain_field',
                     'liz_shell_lmax', 'em_mix_param']  # vector parameters
    # Header
    hline = 80 * '='
    separator = 18 * ' ' + 3 * ('* * *' + 14 * ' ')
    header = [hline, '{:^80s}'.format('Input Parameter Data File'),
              hline, separator, hline,
              '{:^80}'.format('System Related Parameters'), hline]

    natoms = ['No. Atoms in System (> 0)  ::  '
                       + str(len(atoms)), hline, separator, hline]

    # Get number of atoms from CPA sites if self.parameters['method'] == 3
    if 'method' in parameters.keys():
        if parameters['method'] == 3:
            natoms = ['No. Atoms in System (> 0)  ::  '
                       + str(len(atoms.info['CPA'])), hline, separator, hline]

    header += natoms

    with open('i_new', 'w') as filehandle:
        for entry in header:
            filehandle.write('%s\n' % entry)

    # Rest of the parameters:
    contents = []

    for key in parameters.keys():
        if key in energy_params:
            parameters[key] = parameters[key] / Rydberg

        if key in spatial_params:
            parameters[key] = parameters[key] / Bohr

        if key in vector_params:
            parameters[key] = str(parameters[key])[1:-1]

        if key == 'in_pot':
            for index in parameters['in_pot'].keys():
                contents.append(defaults[key] + '  ::  '
                                + index + ' ' + parameters['in_pot'][index])
        else:
            contents.append(defaults[key] + '  ::  ' + str(parameters[key]))

    with open('i_new', 'a') as filehandle:
        for entry in contents:
            filehandle.write('%s\n' % entry)
