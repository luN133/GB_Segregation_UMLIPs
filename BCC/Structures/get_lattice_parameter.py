import numpy as np
from ase.build import bulk
from ase.build import make_supercell
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from BCC.Functions_BCC import get_lattice_parameter, define_calc


def check_minimum(pot):
    a0_Fe = get_lattice_parameter(pot)
    pure_bulk = bulk('Fe',
                     'bcc',
                     a=a0_Fe,
                     cubic=True)
    multiplier = np.identity(3) * 4
    pure_bulk = make_supercell(pure_bulk, multiplier)
    pure_bulk.calc = define_calc(pot, caltype='Fe_only')
    ucf = FrechetCellFilter(pure_bulk, hydrostatic_strain=True)
    opt = BFGS(ucf)
    opt.run(fmax=1e-4)
    a0_Fe = pure_bulk.cell[0][0] / 4
    print(f"The lattice parameter of {pot} is {a0_Fe}")

pot_list = []
for pot in pot_list:
    check_minimum(pot)