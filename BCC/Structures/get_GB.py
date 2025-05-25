from ase.io import read, write
from BCC.Functions_BCC import get_lattice_parameter, get_pot_list_uMLIPs, get_pot_list_EIPs, get_GB
import numpy as np


def scale_factor_GB(pot):
    a0_Fe_Mai = 2.832
    a0_Fe = get_lattice_parameter(pot)
    scale_factor = a0_Fe / a0_Fe_Mai
    return scale_factor


def rescaling_GBs_4uMLIPs(pot, gb):
    GB_file_path = f"GBs/{gb}_Mai.vasp"
    scale_factor = scale_factor_GB(pot)
    with open(GB_file_path, 'r') as file:
        info = file.read()
    new_info = info[:5] + str(scale_factor) + info[8:]
    with open(f"GBs/Rescaled/{pot}_{gb}.vasp", 'w') as new_file:
        new_file.write(new_info)


def rescaling_GBs_4EIPs(pot, gb):
    GB = read(f'GBs/{gb}_Mai.vasp')
    GB_rescaled = GB.copy()
    scale_factor = scale_factor_GB(pot)
    x_new = GB.get_cell()[0][0] * scale_factor
    y_new = GB.get_cell()[1][1] * scale_factor
    z_new = GB.get_cell()[2][2] * scale_factor
    GB_rescaled.set_cell([x_new, y_new, z_new], scale_atoms=True)
    write(f'GBs/Rescaled/{pot}_{gb}.lammps-data', GB_rescaled)


def delete_vac(pot, gb): # used for calculating GB energies
    GB = get_GB(pot, gb)
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
    for pot in pot_list_EIPs:
        write(f'GBs/No_vac_unrelaxed/{pot}_{gb}_novac.lammps-data', GB)
    for pot in pot_list_uMLIPs:
        write(f'GBs/No_vac_unrelaxed/{pot}_{gb}_novac.vasp', GB)



pot_list = []
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)'] #[110]

for pot in pot_list:
    for gb in GB_list:
        rescaling_GBs_4uMLIPs(pot, gb)