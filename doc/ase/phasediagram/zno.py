# creates: zno.png
from ase.phasediagram import solvated
from ase.pourbaix import Pourbaix

refs = solvated('Zn')
refs += [('Zn', 0.0), ('ZnO', -3.323), ('ZnO2(aq)', -2.921)]
pbx = Pourbaix('ZnO', dict(refs))
energy, phase = pbx.get_pourbaix_energy(1.0, 9.0, verbose=True)
print(type(phase))
print(phase.get_free_energy(1.0, 9.0))
pbx.get_pourbaix_energy(0.0, 10.0, verbose=True)
Urange = [-2, 2]
pHrange = [0, 14]
pbx.plot(
    Urange=Urange,
    pHrange=pHrange,
    show=False,
    include_text=True,
    savefig='zno.png'
)
