from ase.io import read, write
from Ito_ASE_setup import get_GB_rescaled, define_calc, set_optimizer, get_pot_list_EIPs, get_pot_list_uMLIPs, get_GB_novac_unrelaxed
import sys
from ase.constraints import UnitCellFilter

def relax_GB(potchoice, GBchoice, optimizer, fmax, steps):
    print(f"\n\nRelaxation {GBchoice}, {potchoice}....\n")
    GB = get_GB_rescaled(potchoice, GBchoice)
    caltype = 'Fe_only'
    GB.calc = define_calc(potchoice, caltype)
    opt = set_optimizer(optimizer, GB)
    opt.run(fmax=fmax, steps=steps)
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_EIPs:
        write(f'Structures/GBs/Relaxed/{potchoice}_{GBchoice}_relaxed.lammps-data', GB)
    if potchoice in pot_list_uMLIPs:
        write(f'Structures/GBs/Relaxed/{potchoice}_{GBchoice}_relaxed.vasp', GB)


def relax_GB_novac(potchoice, GBchoice, optimizer, fmax, steps):
    print(f"\n\nRelaxation {GBchoice}, {potchoice}....\n")
    GB = get_GB_novac_unrelaxed(potchoice, GBchoice)
    caltype = 'Fe_only'
    GB.calc = define_calc(potchoice, caltype)
    ucf = UnitCellFilter(GB,
                         mask = [0, 1, 0, 0, 0, 0], #(xx, yy, zz, yz, xz, xy) True: relax to 0, False: fix
                         hydrostatic_strain=True,
                         cell_factor=float(len(GB)*10))
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_EIPs:
        write(f'Structures/GBs/No_vac_relaxed/{potchoice}_{GBchoice}_relaxed.lammps-data', GB)
    if potchoice in pot_list_uMLIPs:
        write(f'Structures/GBs/No_vac_relaxed/{potchoice}_{GBchoice}_relaxed.vasp', GB)


'''
#pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
pot_list = ['CHGNet(v0_3_0)', 'CHGNet(v0_2_0)']
GB_list = ['Sigma37(610)', 'Sigma13(510)', 'Sigma17(410)', 'Sigma5(310)', 'Sigma29(520)', 'Sigma5(210)', 'Sigma13(320)', 'Sigma25(430)', 'Sigma41(540)']
optimizer = 'FIRE'
fmax = 1e-4
steps = 4000
for potchoice in pot_list:
    for GBchoice in GB_list:
        relax_GB(potchoice, GBchoice, optimizer, fmax)
'''

pot_list_uMLIPs = get_pot_list_uMLIPs()
pot_list_EIPs = get_pot_list_EIPs()

if __name__ == "__main__":
    potchoice = sys.argv[1]
    GBchoice = sys.argv[2]
    if potchoice in pot_list_EIPs:
        steps = 50000
        optimizer = 'FIRE'  # Global optmizer
    if potchoice in pot_list_uMLIPs:
        steps = 5000
        optimizer = 'BFGS'
    fmax = 1e-4  # Global fmax
    relax_GB_novac(potchoice, GBchoice, optimizer, fmax, steps)