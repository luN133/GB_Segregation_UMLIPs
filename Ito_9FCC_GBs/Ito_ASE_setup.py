from ase.io import read
from chgnet.model import CHGNetCalculator
from mace.calculators import mace_mp
from ase.optimize import BFGS, FIRE
from chgnet.model.model import CHGNet
import matgl
from matgl.ext.ase import M3GNetCalculator
import torch
import os
from ase.calculators.lammpslib import LAMMPSlib
from sevenn.sevennet_calculator import SevenNetCalculator


def get_pot_list_uMLIPs():
    pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
    return pot_list

def get_pot_list_EIPs():
    pot_list = ['FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
    return pot_list


def get_pure_bulk(potchoice):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_EIPs:
        pure_bulk = read(f"Structures/Pure_bulks/pure_bulk_{potchoice}.lammps-data")
        pure_bulk.symbols = 'Fe'
    if potchoice in pot_list_uMLIPs:
        pure_bulk = read(f"Structures/Pure_bulks/pure_bulk_{potchoice}.vasp")
    return pure_bulk


def get_GB(potchoice, GBchoice):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_EIPs:
        GB = read(f"Structures/GBs/Relaxed/{potchoice}_{GBchoice}_relaxed.lammps-data")
        GB.symbols = 'Fe'
    if potchoice in pot_list_uMLIPs:
        GB = read(f"Structures/GBs/Relaxed/{potchoice}_{GBchoice}_relaxed.vasp")
    return GB


def get_GB_rescaled(potchoice, GBchoice):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_EIPs:
        GB = read(f"Structures/GBs/Rescaled/{potchoice}_{GBchoice}.lammps-data")
        GB.symbols = 'Fe'
    if potchoice in pot_list_uMLIPs:
        GB = read(f"Structures/GBs/Rescaled/{potchoice}_{GBchoice}.vasp")
    return GB


def get_GB_novac_unrelaxed(potchoice, GBchoice):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_EIPs:
        GB = read(f"Structures/GBs/No_vac_unrelaxed/{potchoice}_{GBchoice}_novac.lammps-data")
        GB.symbols = 'Fe'
    if potchoice in pot_list_uMLIPs:
        GB = read(f"Structures/GBs/No_vac_unrelaxed/{potchoice}_{GBchoice}_novac.vasp")
    return GB


def get_GB_novac_relaxed(potchoice, GBchoice):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_EIPs:
        GB = read(f"Structures/GBs/No_vac_relaxed/{potchoice}_{GBchoice}_relaxed.lammps-data")
        GB.symbols = 'Fe'
    if potchoice in pot_list_uMLIPs:
        GB = read(f"Structures/GBs/No_vac_relaxed/{potchoice}_{GBchoice}_relaxed.vasp")
    return GB


def get_bulk_number_of_atom(potchoice):
    pure_bulk = get_pure_bulk(potchoice)
    bulk_number_of_atom = pure_bulk.get_global_number_of_atoms()
    return bulk_number_of_atom


def get_GB_number_of_atom(potchoice, GBchoice):
    GB = get_GB(potchoice, GBchoice)
    GB_number_of_atom = GB.get_global_number_of_atoms()
    return GB_number_of_atom


def get_cal_sites(potchoice, GBchoice):
    GB = get_GB(potchoice, GBchoice)
    lower_b = 12.4  # Define the lower boundary 2.5 angstrom from the GB plane at y = 15, tolerance = 0.1
    upper_b = 15.1  # Define the upper boundary. Only one side from the GB, it's equivalent
    cal_sites = []
    for index, atom in enumerate(GB):  # Iterate through the atoms in the GB structure
        # Check if the z-coordinate of the atom is within the specified range
        if lower_b <= atom.position[1] <= upper_b:
            # If it is, store the index of the atom
            cal_sites.append(index)
    return cal_sites


def get_lattice_parameter(potchoice):
    a0_Cu_Tschopp = 3.615
    if potchoice == 'MACE':
        a0_Fe = 3.6335
    elif potchoice == 'CHGNet(v0_3_0)':
        a0_Fe = 3.4956
    elif potchoice == 'CHGNet(v0_2_0)':
        a0_Fe = 3.4628
    elif potchoice == 'M3GNet':
        a0_Fe = 3.463890989662329
    elif potchoice == 'SevenNet':
        a0_Fe = 3.5164437099676644
    elif potchoice == 'FeMnNiCu_Bonny':
        a0_Fe = 3.6582108972171676
    elif potchoice == 'FeCrCoNiCu_Deluigi':
        a0_Fe = 3.5600977277879
    elif potchoice == 'FeTiCoNiCuMoW_Zhou':
        a0_Fe = 3.6284420233011447
    elif potchoice == 'FeCu_Lee':
        a0_Fe = 3.611098367149702
    else:
        a0_Fe = a0_Cu_Tschopp
    return a0_Fe


def define_calc(potchoice, caltype=None, sub_e_calc=None):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if potchoice in pot_list_uMLIPs:
        if potchoice == 'MACE':
            calc = mace_mp(model='large', default_dtype='float64')
        elif potchoice == 'CHGNet(v0_3_0)':
            torch.set_default_dtype(torch.float32)
            calc = CHGNetCalculator(CHGNet.load('0.3.0'))
        elif potchoice == 'CHGNet(v0_2_0)':
            torch.set_default_dtype(torch.float32)
            calc = CHGNetCalculator(CHGNet.load('0.2.0'))
        elif potchoice == 'M3GNet':
            torch.set_default_dtype(torch.float32)
            calc = M3GNetCalculator(matgl.load_model('M3GNet-MP-2021.2.8-PES'))
        elif potchoice == 'SevenNet':
            calc = SevenNetCalculator("7net-0", device='cpu')
    elif potchoice in pot_list_EIPs:
        list_EAM = ['FeMnNiCu_Bonny', 'FeTiCoNiCuMoW_Zhou', 'FeCrCoNiCu_Deluigi']
        list_MEAM = ['FeCu_Lee']
        if caltype == 'Fe_only':
            if potchoice in list_EAM:
                cmds = ["pair_style eam/alloy",
                        f"pair_coeff * * IPs/{potchoice}.eam.alloy Fe"]
            if potchoice in list_MEAM:
                if potchoice == 'FeCu_Lee':
                    cmds = ["pair_style meam",
                            f"pair_coeff * * IPs/library_FeCu_Lee.meam Fe Cu IPs/FeCu_Lee.meam Fe"]
            atom_types = {'Fe': 1}
        if caltype == 'sub':
            if potchoice in list_EAM:
                cmds = ["pair_style eam/alloy",
                        f"pair_coeff * * IPs/{potchoice}.eam.alloy Fe {sub_e_calc}"]
            if potchoice in list_MEAM:
                if potchoice == 'FeCu_Lee':
                    cmds = ["pair_style meam",
                            f"pair_coeff * * IPs/library_FeCu_Lee.meam Fe Cu IPs/FeCu_Lee.meam Fe {sub_e_calc}"]
            atom_types = {f'{sub_e_calc}': 2, 'Fe': 1}
        calc = LAMMPSlib(lmpcmds=cmds, atom_types=atom_types)
    return calc


def set_optimizer(optimizer, structure):
    if optimizer == 'BFGS':
        opt = BFGS(structure)
    if optimizer == 'FIRE':
        opt = FIRE(structure)
    return opt


def get_e_list(potchoice):
    #e_list = ['Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo']
    pot_list_uMLIPs = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
    pot_list_EIPs = ['FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
    if potchoice in pot_list_uMLIPs:
        e_list = ['Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo']
    if potchoice in pot_list_EIPs:
        if potchoice == 'FeMnNiCu_Bonny':
            e_list = ['Mn', 'Ni', 'Cu']
        if potchoice == 'FeTiCoNiCuMoW_Zhou':
            e_list = ['Ti', 'Co', 'Ni', 'Cu', 'Mo']
        if potchoice == 'FeCu_Lee':
            e_list = ['Cu']
        if potchoice == 'FeCrCoNiCu_Deluigi':
            e_list = ['Cr', 'Co', 'Ni', 'Cu']
    return e_list


def create_needed_dir(pot_list):
    path_results = 'Results'
    os.makedirs(path_results, exist_ok=True)
    path_Details = 'Results/Details'
    os.makedirs(path_Details, exist_ok=True)
    path_E_seg_min = 'Results/E_seg_min'  # Folders to save the results of E_seg
    os.makedirs(path_E_seg_min, exist_ok=True)
    path_traj = 'Results/Trajectories'
    os.makedirs(path_traj, exist_ok=True)
    for potchoice in pot_list:
        path_individual_pot = f'Results/Details/{potchoice}'
        os.makedirs(path_individual_pot, exist_ok=True)



