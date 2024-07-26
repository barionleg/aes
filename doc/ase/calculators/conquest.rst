.. module:: ase.calculators.conquest

========
Conquest
========

.. image:: ../../static/conquest.png

`Conquest <http://http://www.order-n.org>`_ is density functional theory
electronic structure code including periodic boundary conditions. Conquest is
based on the pseudo-potential approximation where the Kohn-Sham wavefunctions 
of valence electrons are expanded onto a strictly localized numerical 
atomic-centered basis, also referred to as pseudo-atomic orbitals (PAOs).
Conquest is designed to scale to large systems, either using exact 
diagonalisation or with linear scaling. *Note that the current version of the 
ASE calculator is restricted to diagonalisation.* Implemented ASE properties
are ``energy``, ``forces`` and ``stress``. The Conquest calculator can also be 
used to manage geometry optimisation and molecular dynamics. A more detailed
description of the calculator possibilities is available in the 
`ASE/Conquest documentation <https://conquest.readthedocs.io/en/latest/ext-tools.html#atomic-simulation-environment-ase>`_.


Setup
=====

Environment variables
---------------------

* ``ASE_CONQUEST_COMMAND``: the Conquest executable command including MPI/openMPI prefix.
* ``CQ_PP_PATH``: the PAO path directory to where are located the the ``.ion`` files.

Given the Conquest root directory ``CQ_ROOT``, initialisation might look to something like::

    import os

    CQ_ROOT = 'PATH_TO_CONQUEST_DIRECTORY'
    os.environ['ASE_CONQUEST_COMMAND'] = 'mpirun -np 4 '+CQ_ROOT+'/src/Conquest'
    os.environ['CQ_PP_PATH'] = CQ_ROOT+'/pseudo-and-pao/'

Pseudopotential/PAO basis
--------------------------

For each element, pseudopotential and numerical basis functions are collected 
in a single file having the extension ``.ion``. The pseudopotential/basis files 
are specified through a python dictionary, for example::

    basis = {'O' : {'file' : 'O_SZP.ion'},
             'H' : {'file' : 'H_SZP.ion'},
             'C' : {'file' : 'C_SZP.ion'}}


Conquest Calculator Class
=========================

.. autoclass:: ase.calculators.conquest.Conquest

Minimal parameters calculation
==============================

Having set up the environment variables properly, the minimal set of parameters 
to launch a SCF calculation is given below with the example of NaCl::

    struct = bulk('NaCl', crystalstructure='rocksalt', a=5.71, cubic=True)    
    basis  = {'Cl' : {'file' : 'ClCQ.ion'}, 'Na' : {'file' : 'NaCQ.ion'}}    
    
    struct.calc = Conquest(basis=basis)
    struct.get_potential_energy()

Default parameters are given in the ``Calculator Class`` above. Note that:

* When ``kpts`` is set to ``None``, only the Γ-point is considered for the calculation.
* The calculation files will be located in the current directory.

Parameters
==========

In principle all the `Conquest input parameters <https://conquest.readthedocs.io/en/latest/input_tags.html>`_
can be managed by the calculator using key/value pairs in a dictionary. Below is
a list of other *important* parameters set as default by the Calculator.

===============================  =========  ===============  ================================
keyword                          type       default value    description
===============================  =========  ===============  ================================
``IO.WriteOutToASEFile``         ``bool``   True             must always be True when using ASE
``DM.SolutionMethod``            ``str``    'diagon'         must always be 'diagon' when using ASE
``IO.Iprint``                    ``int``    1                verbose for the output
``General.PseudopotentialType``  ``str``    'Hamann'         kind of pseudopotential other type are 'siesta' and 'abinit'
``SC.MaxIters``                  ``int``    50               maximum number SCF cycles
``AtomMove.TypeOfRun``           ``str``    'static'         'static' stands for single (non)SCF other are 'md' or optimisation algorithms.
``Diag.SmearingType``            ``int``    1                1 for Methfessel-Paxton ; 0 for Fermi-Dirac
``Diag.kT``                      ``float``  0.001            smearing temperature in Ha
===============================  =========  ===============  ================================

..
  ``io.fractionalatomiccoords``    ``bool``   True             atomic coordinates format for the structure file (fractional or cartesian)
  ``basis.basisset``               ``str``    'PAOs'           type of basis set ; always 'PAOs' with ASE 

An example of more advanced Calculator setup is given below for a SCF calculation on Na::

    struct = bulk('Na', crystalstructure='bcc', a=4.17, cubic=True)    
    basis  = {'Na' : {'file' : 'NaCQ.ion'}}
    
    conquest_flags = {'Diag.SmearingType': 0,
                      'Diag.kT'          : 0.005}
                      
    struct.calc = Conquest(directory      = 'Na_bcc_example',
                           grid_cutoff    = 90.0,
                           self_consistent= True,
                           xc    = 'PBE',
                           basis = basis,
                           kpts  = [6,6,6],
                           nspin = 1,
                           **conquest_flags)
    
    struct.get_potential_energy()


Any ``FixAtoms`` constraints are converted to Conquest Cartesian constraints
when running geometry optimisation or molecular dynamic calculations.



