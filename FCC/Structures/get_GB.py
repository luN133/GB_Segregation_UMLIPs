from ase.io import write, read
import numpy as np
from ase.build import make_supercell
from FCC.Functions_FCC import get_lattice_parameter, get_pot_list_EIPs, get_pot_list_uMLIPs, get_GB


def tailor2ItoGB(gb):
    GB = read(f"GBs/Original/{gb}_origin.lammps-data", atom_style='atomic')
    # Ito's GBs have a length of 2 a0_Fe along y direction
    multiplier = np.identity(3)
    multiplier[2][2] = 2
    GB = make_supercell(GB, multiplier)
    # Delete atoms from top and bottom such that the atom number of GB matches Ito's paper
    y_coordinates = GB.positions[:, 1]
    sorted_indices = np.argsort(y_coordinates)
    # Need to be specified manually
    if gb == 'Sigma37(610)':
        indices_to_delete = sorted_indices[:1632].tolist() + sorted_indices[-1612:].tolist()
    elif gb == 'Sigma13(510)':
        indices_to_delete = sorted_indices[:1348].tolist() + sorted_indices[-1332:].tolist()
    elif gb == 'Sigma17(410)':
        indices_to_delete = sorted_indices[:1140].tolist() + sorted_indices[-1128:].tolist()
    elif gb == 'Sigma5(310)':
        indices_to_delete = sorted_indices[:792].tolist() + sorted_indices[-824:].tolist()
    elif gb == 'Sigma29(520)':
        indices_to_delete = sorted_indices[:1512].tolist() + sorted_indices[-1496:].tolist()
    elif gb == 'Sigma5(210)':
        indices_to_delete = sorted_indices[:532].tolist() + sorted_indices[-568:].tolist()
    elif gb == 'Sigma13(320)':
        indices_to_delete = sorted_indices[:964].tolist() + sorted_indices[-956:].tolist()
    elif gb == 'Sigma25(430)':
        indices_to_delete = sorted_indices[:1292].tolist() + sorted_indices[-1280:].tolist()
    elif gb == 'Sigma41(540)':
        indices_to_delete = sorted_indices[:1832].tolist() + sorted_indices[-1816:].tolist()
    else:
        raise ValueError("GB is not included in Ito's paper.")
    GB_new = GB.copy()
    del GB_new[[indices_to_delete]]
    # Set the cell dimension to 30 as the one in Ito's paper and center the grain boundary within the new cell size
    y_min = np.min(GB_new.positions[:, 1])
    y_max = np.max(GB_new.positions[:, 1])
    y_center = (y_max + y_min) / 2  # Calculate the position of GB plane
    shift = 15 - y_center
    GB_new.positions[:, 1] += shift  # Apply the shift to the atomic positions
    GB_new.cell[1][1] = 30  # Set the cell size in z-axis to 30 without rescaling
    # Rescale the cell dimension in the other two axis with rescaling of atoms
    GB.set_pbc((True, False, True))
    if gb == 'Sigma37(610)':
        GB_new.set_cell([21.53, 30, 7.08], scale_atoms=True)
    elif gb == 'Sigma13(510)':
        GB_new.set_cell([18.05, 30, 7.08], scale_atoms=True)
    elif gb == 'Sigma17(410)':
        GB_new.set_cell([14.60, 30, 7.08], scale_atoms=True)
    elif gb == 'Sigma5(310)':
        GB_new.set_cell([11.19, 30, 7.08], scale_atoms=True)
    elif gb == 'Sigma29(520)':
        GB_new.set_cell([19.06, 30, 7.08], scale_atoms=True)
    elif gb == 'Sigma5(210)':
        GB_new.set_cell([7.92, 30, 7.08], scale_atoms=True)
    elif gb == 'Sigma13(320)':
        GB_new.set_cell([12.76, 30, 7.08], scale_atoms=True)
    elif gb == 'Sigma25(430)':
        GB_new.set_cell([17.70, 30, 7.08], scale_atoms=True)
    elif gb == 'Sigma41(540)':
        GB_new.set_cell([22.67, 30, 7.08], scale_atoms=True)
    # Replace all the atoms with Fe
    number_of_atom_GB = GB_new.get_global_number_of_atoms()
    GB_new.symbols = 'Fe'
    write(f"GBs/Tailored/{gb}_tailored2Ito.vasp", GB_new)


def scale_factor_GB(pot):
    a0_Cu_Tschopp = 3.615
    a0_Fe = get_lattice_parameter(pot)
    return a0_Fe / a0_Cu_Tschopp


def rescaling_GBs(pot, gb):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    GB = read(f'GBs/Tailored/{gb}_tailored2Ito.vasp')
    GB_rescaled = GB.copy()
    scale_factor = scale_factor_GB(pot)
    x_new = GB.get_cell()[0][0] * scale_factor
    z_new = GB.get_cell()[2][2] * scale_factor
    GB_rescaled.set_cell([x_new, 30, z_new], scale_atoms=True)
    if pot in pot_list_EIPs:
        write(f'GBs/Rescaled/{pot}_{gb}.lammps-data', GB_rescaled)
    if pot in pot_list_uMLIPs:
        write(f'GBs/Rescaled/{pot}_{gb}.vasp', GB_rescaled)


def delete_vac(pot, gb):
    GB = get_GB(pot, gb)
    positions = GB.get_positions()
    y_min = np.min(positions[:, 1])
    y_max = np.max(positions[:, 1])
    new_y_length = y_max - y_min + 2
    cell = GB.get_cell()
    cell[1, 1] = new_y_length
    GB.set_cell(cell, scale_atoms=False)
    positions[:, 1] -= y_min
    GB.set_positions(positions)
    GB.wrap()


    pot_list_EIPs = get_pot_list_EIPs()
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    for pot in pot_list_EIPs:
        write(f'GBs/No_vac_unrelaxed/{pot}_{gb}_novac.lammps-data', GB)
    for pot in pot_list_uMLIPs:
        write(f'GBs/No_vac_unrelaxed/{pot}_{gb}_novac.vasp', GB)



pot_list = []
GB_list = ['Sigma37(610)', 'Sigma13(510)', 'Sigma17(410)', 'Sigma5(310)', 'Sigma29(520)',
           'Sigma5(210)', 'Sigma13(320)', 'Sigma25(430)', 'Sigma41(540)']

for gb in GB_list:
    tailor2ItoGB(gb)
for pot in pot_list:
    for gb in GB_list:
        rescaling_GBs(pot, gb)





