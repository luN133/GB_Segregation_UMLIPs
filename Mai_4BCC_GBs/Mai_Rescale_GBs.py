from ase.io import read, write
from Mai_ASE_setup import get_lattice_parameter, get_pot_list_uMLIPs, get_pot_list_EIPs, get_GB
import numpy as np




def scale_factor_GB(potchoice):
    a0_Fe_Mai = 2.832
    a0_Fe = get_lattice_parameter(potchoice)
    scale_factor = a0_Fe / a0_Fe_Mai
    return scale_factor


def rescaling_GBs1(potchoice, GBchoice):
    file_path = f"Structures/{GBchoice}_Mai.vasp"
    scale_factor = scale_factor_GB(potchoice)
    with open(file_path, 'r') as file:
        content = file.read()
    new_content = content[:5] + str(scale_factor) + content[8:]
    with open(f"Structures/GBs/Rescaled/{potchoice}_{GBchoice}.vasp", 'w') as new_file:
        new_file.write(new_content)


def rescaling_GBs2(potchoice, GBchoice):
    GB = read(f'Structures/{GBchoice}_Mai.vasp')
    GB_rescaled = GB.copy()
    scale_factor = scale_factor_GB(potchoice)
    x_new = GB.get_cell()[0][0] * scale_factor
    y_new = GB.get_cell()[1][1] * scale_factor
    z_new = GB.get_cell()[2][2] * scale_factor
    GB_rescaled.set_cell([x_new, y_new, z_new], scale_atoms=True)
    write(f'Structures/GBs/Rescaled/{potchoice}_{GBchoice}.lammps-data', GB_rescaled)


def delete_vac(potchoice, GBchoice):
    GB = get_GB(potchoice, GBchoice)
    positions = GB.get_positions()
    z_min = np.min(positions[:, 2])
    z_max = np.max(positions[:, 2])
    new_z_length = z_max - z_min + 2
    cell = GB.get_cell()
    cell[2, 2] = new_z_length
    GB.set_cell(cell, scale_atoms=False)
    positions[:, 2] -= z_min
    GB.set_positions(positions)
    GB.wrap()

    pot_list_EIPs = get_pot_list_EIPs()
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    for potchoice in pot_list_EIPs:
        write(f'Structures/GBs/No_vac_unrelaxed/{potchoice}_{GBchoice}_novac.lammps-data', GB)
    for potchoice in pot_list_uMLIPs:
        write(f'Structures/GBs/No_vac_unrelaxed/{potchoice}_{GBchoice}_novac.vasp', GB)


pot_list1 = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)'] #[110]
pot_list2 = ['FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']

for potchoice in pot_list1:
    for GBchoice in GB_list:
        rescaling_GBs1(potchoice, GBchoice)
for potchoice in pot_list2:
    for GBchoice in GB_list:
        rescaling_GBs2(potchoice, GBchoice)

pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
for potchoice in pot_list:
    for GBchoice in GB_list:
        delete_vac(potchoice, GBchoice)
