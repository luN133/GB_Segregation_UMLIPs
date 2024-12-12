from ase.io import write
from ase.build import bulk
import numpy as np
from ase.build import make_supercell
from Mai_ASE_setup import get_lattice_parameter

pot_list1 = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
pot_list2 = ['FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']


'''
def multiplier4bulk(GBchoice):
    multiplier = np.identity(3)
    if GBchoice == 'Sigma3(1-11)[110]': # atom number = 72
        multiplier[0][0] = 3
        multiplier[1][1] = 3
        multiplier[2][2] = 4
    if GBchoice == 'Sigma3(1-12)[110]': # atom number = 48
        multiplier[0][0] = 2
        multiplier[1][1] = 2
        multiplier[2][2] = 6
    if GBchoice == 'Sigma9(2-21)[110]': # atom number = 68
        multiplier[0][0] = 1
        multiplier[1][1] = 2
        multiplier[2][2] = 17
    if GBchoice == 'Sigma11(3-32)[110]': # atom number = 42
        multiplier[0][0] = 1
        multiplier[1][1] = 3
        multiplier[2][2] = 7
    return multiplier
'''

def create_pure_bulk1(potchoice):
    a0_Fe = get_lattice_parameter(potchoice)
    pure_bulk = bulk('Fe',
                     'bcc',
                     a=a0_Fe,
                     cubic=True)
    multiplier = np.identity(3)*4
    pure_bulk = make_supercell(pure_bulk, multiplier)
    write(f"Structures/Pure_bulks/pure_bulk_{potchoice}.vasp", pure_bulk)


def create_pure_bulk2(potchoice):
    a0_Fe = get_lattice_parameter(potchoice)
    pure_bulk = bulk('Fe',
                     'bcc',
                     a=a0_Fe,
                     cubic=True)
    multiplier = np.identity(3)*4
    pure_bulk = make_supercell(pure_bulk, multiplier)
    write(f"Structures/Pure_bulks/pure_bulk_{potchoice}.lammps-data", pure_bulk)


for potchoice in pot_list1:
    create_pure_bulk1(potchoice)
for potchoice in pot_list2:
    create_pure_bulk2(potchoice)





