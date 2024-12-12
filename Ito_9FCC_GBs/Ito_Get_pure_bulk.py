from ase.io import write
from ase.build import bulk
import numpy as np
from ase.build import make_supercell
import os
from Ito_ASE_setup import get_lattice_parameter, get_pot_list_EIPs, get_pot_list_uMLIPs



def get_pure_bulk(potchoice):
    a0_Fe = get_lattice_parameter(potchoice)
    pure_bulk = bulk('Fe',
                     'fcc',
                     a=a0_Fe,
                     cubic=True)
    multiplier = np.identity(3)*3
    pure_bulk = make_supercell(pure_bulk, multiplier)
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_uMLIPs:
        write(f"Structures/Pure_bulks/pure_bulk_{potchoice}.vasp", pure_bulk)
    if potchoice in pot_list_EIPs:
        write(f"Structures/Pure_bulks/pure_bulk_{potchoice}.lammps-data", pure_bulk)



def check_path_dirs():
    path_Pure_bulks = 'Structures/Pure_bulks'
    if not os.path.exists(path_Pure_bulks):
        os.makedirs(path_Pure_bulks)


pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet',  'SevenNet', 'FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
check_path_dirs()
for potchoice in pot_list:
    get_pure_bulk(potchoice)
