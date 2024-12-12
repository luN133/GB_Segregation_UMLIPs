from chgnet.model import CHGNetCalculator
from mace.calculators import mace_mp
from ase.optimize import BFGS, FIRE
import numpy as np
from chgnet.model.model import CHGNet
import matgl
from matgl.ext.ase import M3GNetCalculator
import torch
from ase.build import bulk, make_supercell
from sevenn.sevennet_calculator import SevenNetCalculator
from ase.calculators.lammpslib import LAMMPSlib



def get_pot_list_uMLIPs():
    pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
    return pot_list

def get_pot_list_EIPs():
    pot_list = ['FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
    return pot_list


def get_lattice_parameter(potchoice):
    a0_Fe_Mai = 2.832
    if potchoice == 'MACE':
        a0_Fe = 2.8592
    elif potchoice == 'CHGNet(v0_3_0)':
        a0_Fe = 2.8472
    elif potchoice == 'CHGNet(v0_2_0)':
        a0_Fe = 2.8489
    elif potchoice == 'M3GNet':
        a0_Fe = 2.8521164288520793
    elif potchoice == 'SevenNet':
        a0_Fe = 2.8456892365576505
    elif potchoice == 'FeMnNiCu_Bonny':
        a0_Fe = 2.8553245625400985
    elif potchoice == 'FeCrCoNiCu_Deluigi':
        a0_Fe = 2.860321376074101
    elif potchoice == 'FeTiCoNiCuMoW_Zhou':
        a0_Fe = 2.865951891551427
    elif potchoice == 'FeCu_Lee':
        a0_Fe = 2.8637127695447426
    else:
        a0_Fe = a0_Fe_Mai
    return a0_Fe


def build_Fe54(potchoice):
    a0_Fe = get_lattice_parameter(potchoice)
    Fe54 = bulk('Fe', 'bcc', a=a0_Fe, cubic=True)
    multiplier = np.identity(3) * 3
    Fe54 = make_supercell(Fe54, multiplier)
    return Fe54


def set_optimizer(optimizer, structure):
    if optimizer == 'BFGS':
        opt = BFGS(structure)
    if optimizer == 'FIRE':
        opt = FIRE(structure)
    return opt


def define_calc(potchoice, caltype=None, sub_e_calc=None):
    pot_list_uMLIPs = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
    pot_list_EIPs = ['FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
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

