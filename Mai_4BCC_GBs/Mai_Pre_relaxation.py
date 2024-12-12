from ase.io import read, write
from Mai_ASE_setup import get_GB_rescaled, define_calc, set_optimizer, get_pot_list_EIPs,get_pot_list_uMLIPs, get_GB_novac_unrelaxed
from ase.constraints import UnitCellFilter
import sys


def relax_GB(potchoice, GBchoice, optimizer, fmax, steps):
    print(f"\n\nRelaxation {GBchoice}, {potchoice}....\n")
    GB = get_GB_rescaled(potchoice, GBchoice)
    caltype = 'Fe_only'
    GB.calc = define_calc(potchoice, caltype)
    '''
    ucf = UnitCellFilter(GB,
                         mask = [1, 0, 1, 0, 0, 0], #(xx, yy, zz, yz, xz, xy)
                         hydrostatic_strain=True,
                         cell_factor=float(len(GB)*10))
    '''
    opt = set_optimizer(optimizer, GB)
    opt.run(fmax=fmax, steps=steps)
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_uMLIPs:
        write(f'Structures/GBs/Relaxed/{potchoice}_{GBchoice}_relaxed.vasp', GB)
    if potchoice in pot_list_EIPs:
        write(f'Structures/GBs/Relaxed/{potchoice}_{GBchoice}_relaxed.lammps-data', GB)


def relax_GB_novac(potchoice, GBchoice, optimizer, fmax, steps):
    print(f"\n\nRelaxation {GBchoice}, {potchoice}....\n")
    GB = get_GB_novac_unrelaxed(potchoice, GBchoice)
    caltype = 'Fe_only'
    GB.calc = define_calc(potchoice, caltype)
    ucf = UnitCellFilter(GB,
                         mask = [0, 0, 1, 0, 0, 0], #(xx, yy, zz, yz, xz, xy) True: relax to 0, False: fix
                         hydrostatic_strain=True,
                         cell_factor=float(len(GB)*10))
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_uMLIPs:
        write(f'Structures/GBs/No_vac_relaxed/{potchoice}_{GBchoice}_relaxed.vasp', GB)
    if potchoice in pot_list_EIPs:
        write(f'Structures/GBs/No_vac_relaxed/{potchoice}_{GBchoice}_relaxed.lammps-data', GB)

'''
pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']
optimizer = 'FIRE'
pot_list_EIPs = get_pot_list_EIPs()
fmax = 1e-4
for potchoice in pot_list:
    if potchoice in pot_list_EIPs:
        steps = 10000
    else:
        steps = 5000
    for GBchoice in GB_list:
        relax_GB_novac(potchoice, GBchoice, optimizer, fmax, steps)
'''

if __name__ == "__main__":
    potchoice = sys.argv[1]
    GBchoice = sys.argv[2]
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_EIPs:
        optimizer = 'FIRE'
        steps = 50000
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    if potchoice in pot_list_uMLIPs:
        optimizer = 'BFGS'
        steps = 5000
    fmax = 1e-4
    relax_GB_novac(potchoice, GBchoice, optimizer, fmax, steps)