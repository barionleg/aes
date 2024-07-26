import pytest
from ase.calculators.pwmat import PWmat, PWmatProfile
from ase.build import bulk


@pytest.mark.skip(reason='1. Users may not compile PWmat \
or no GPU in test environment.\n \
2. It is necessary to submit the job to the computing cluster \
through the SLURM platform using sbatch command! \
3. Pseudopotential files are required.')
def test_main():
    lattice_fcc = {'Fe': 3.43}
    input_data = {
        'Parallel': [1, 4],
        'JOB': 'SCF',
        'IN.ATOM': 'atom.config',
        'IN.PSP': ['Fe-sp.PD04.PBE.UPF'],
        'XCFUNCTIONAL': 'PBE',
    }
    atoms = bulk('Fe', crystalstructure='fcc', a=lattice_fcc['Fe'], cubic=True)
    profile = PWmatProfile(pseudo_dir='.')
    calc = PWmat(profile=profile, input_data=input_data, kspacing=0.04)
    atoms.calc = calc
    atoms.get_potential_energy()
