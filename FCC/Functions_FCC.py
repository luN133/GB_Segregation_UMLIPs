import os
import numpy as np
import csv
import torch
from ase.io import read, write
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS, FIRE
try:
    from mace.calculators import mace_mp
except ImportError:
    pass
try:
    from sevenn.calculator import SevenNetCalculator
except ImportError:
    pass
try:
    from orb_models.forcefield import pretrained
    from orb_models.forcefield.calculator import ORBCalculator
except ImportError:
    pass
try:
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    from tensorpotential.calculator import grace_fm
except ImportError:
    pass
try:
    from fairchem.core import OCPCalculator
except ImportError:
    pass
try:
    from pynep.calculate import NEP
except ImportError:
    pass


##########
def get_pot_list_uMLIPs():
    pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'MACE_mpa_0', 'MACE_omat_0',
                'SevenNet_MF_ompa', 'ORB_omat', 'ORB_mpa', 'GRACE_2L_oam', 'GRACE_2L_omat', 'GRACE_FS_oam',
                'GRACE_FS_omat', 'eqV2_31M_omat_mp_salex', 'NEP_89']
    return pot_list


def get_pot_list_EIPs():
    pot_list = ['FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
    return pot_list


def get_pure_bulk(pot):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_EIPs:
        pure_bulk = read(f"Structures/Pure_bulks/pure_bulk_{pot}.lammps-data")
        pure_bulk.symbols = 'Fe'
    elif pot in pot_list_uMLIPs:
        pure_bulk = read(f"Structures/Pure_bulks/pure_bulk_{pot}.vasp")
        return pure_bulk
    return pure_bulk


def get_GB(pot, gb):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_EIPs:
        GB = read(f"Structures/GBs/Relaxed/{pot}_{gb}_relaxed.lammps-data")
        GB.symbols = 'Fe'
    elif pot in pot_list_uMLIPs:
        GB = read(f"Structures/GBs/Relaxed/{pot}_{gb}_relaxed.vasp")
    return GB


def get_GB_rescaled(pot, gb):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_EIPs:
        GB = read(f"Structures/GBs/Rescaled/{pot}_{gb}.lammps-data")
        GB.symbols = 'Fe'
    elif pot in pot_list_uMLIPs:
        GB = read(f"Structures/GBs/Rescaled/{pot}_{gb}.vasp")
    return GB


def get_GB_novac_unrelaxed(pot, gb):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_EIPs:
        GB = read(f"Structures/GBs/No_vac_unrelaxed/{pot}_{gb}_novac.lammps-data")
        GB.symbols = 'Fe'
    elif pot in pot_list_uMLIPs:
        GB = read(f"Structures/GBs/No_vac_unrelaxed/{pot}_{gb}_novac.vasp")
    return GB


def get_GB_novac_relaxed(pot, gb):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_EIPs:
        GB = read(f"Structures/GBs/No_vac_relaxed/{pot}_{gb}_relaxed.lammps-data")
        GB.symbols = 'Fe'
    elif pot in pot_list_uMLIPs:
        GB = read(f"Structures/GBs/No_vac_relaxed/{pot}_{gb}_relaxed.vasp")
    return GB


def get_bulk_number_of_atom(pot):
    pure_bulk = get_pure_bulk(pot)
    bulk_number_of_atom = pure_bulk.get_global_number_of_atoms()
    return bulk_number_of_atom


def get_GB_number_of_atom(pot, gb):
    GB = get_GB(pot, gb)
    GB_number_of_atom = GB.get_global_number_of_atoms()
    return GB_number_of_atom


def get_cal_sites(pot, gb):
    GB = get_GB(pot, gb)
    lower_b = 12.4  # Define the lower boundary 2.5 angstrom from the GB plane at y = 15, tolerance = 0.1
    upper_b = 15.1  # Define the upper boundary. Only one side from the GB, it's equivalent
    cal_sites = []
    for index, atom in enumerate(GB):  # Iterate through the atoms in the GB structure
        # Check if the z-coordinate of the atom is within the specified range
        if lower_b <= atom.position[1] <= upper_b:
            # If it is, store the index of the atom
            cal_sites.append(index)
    return cal_sites


# To determine coordinate of the GB plane.
def get_GB_plane(pot, gb):
    y_GB_plane = 15
    return y_GB_plane


def get_lattice_parameter(pot):
    a0_Cu_Tschopp = 3.615
    if pot == 'MACE':
        a0_Fe = 3.6335
    elif pot == 'MACE_mpa_0':
        a0_Fe = 3.4532625450700576
    elif pot == 'MACE_omat_0':
        a0_Fe = 3.3848457993063534
    elif pot == 'CHGNet(v0_3_0)':
        a0_Fe = 3.4956
    elif pot == 'CHGNet(v0_2_0)':
        a0_Fe = 3.4628
    elif pot == 'M3GNet':
        a0_Fe = 3.463890989662329
    elif pot == 'SevenNet':
        a0_Fe = 3.5164437099676644
    elif pot == 'SevenNet_MF_ompa':
        a0_Fe = 3.5460874121700976
    elif pot == 'GRACE_2L_oam':
        a0_Fe = 3.490863419158585
    elif pot == 'GRACE_2L_omat':
        a0_Fe = 3.4937777264589074
    elif pot == 'GRACE_FS_oam':
        a0_Fe = 3.590156544708518
    elif pot == 'GRACE_FS_omat':
        a0_Fe = 3.5817113917833403
    elif pot == 'ORB_omat':
        a0_Fe = 3.5009755027607294
    elif pot == 'ORB_mpa':
        a0_Fe = 3.5063559858597166
    elif pot == 'FeMnNiCu_Bonny':
        a0_Fe = 3.6582108972171676
    elif pot == 'FeCrCoNiCu_Deluigi':
        a0_Fe = 3.5600977277879
    elif pot == 'FeTiCoNiCuMoW_Zhou':
        a0_Fe = 3.6284420233011447
    elif pot == 'FeCu_Lee':
        a0_Fe = 3.611098367149702
    elif pot == 'eqV2_31M_omat_mp_salex':
        a0_Fe = 3.5780541328629756
    elif pot == 'NEP_89':
        a0_Fe = 3.357633841546921
    else:
        a0_Fe = a0_Cu_Tschopp
    return a0_Fe


def define_calc(pot, caltype=None, sub_e_calc=None):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_uMLIPs:
        if pot == 'MACE':
            calc = mace_mp(model='large', default_dtype='float64')
        elif pot == 'MACE_mpa_0':
            calc = mace_mp(model='medium-mpa-0', default_dtype='float64')
        elif pot == 'MACE_omat_0':
            calc = mace_mp(model='medium-omat-0', default_dtype='float64')
        elif pot == 'CHGNet(v0_3_0)':
            torch.set_default_dtype(torch.float32)
            calc = CHGNetCalculator(CHGNet.load('0.3.0'))
        elif pot == 'CHGNet(v0_2_0)':
            torch.set_default_dtype(torch.float32)
            calc = CHGNetCalculator(CHGNet.load('0.2.0'))
        elif pot == 'M3GNet':
            torch.set_default_dtype(torch.float32)
            calc = M3GNetCalculator(matgl.load_model('M3GNet-MP-2021.2.8-PES'))
        elif pot == 'SevenNet':
            calc = SevenNetCalculator("7net-0", device='cpu')
        elif pot == 'SevenNet_MF_ompa':
            calc = SevenNetCalculator('7net-mf-ompa', modal='mpa')
        elif pot == 'ORB_omat':
            torch._dynamo.config.suppress_errors = True
            device = 'cpu'
            orbff = pretrained.orb_v3_conservative_inf_omat(
                device=device,
                precision="float64")
            calc = ORBCalculator(orbff, device=device)
        elif pot == 'ORB_mpa':
            torch._dynamo.config.suppress_errors = True
            device = 'cpu'
            orbff = pretrained.orb_v3_conservative_inf_mpa(
                device=device,
                precision="float64")
            calc = ORBCalculator(orbff, device=device)
        elif pot == 'GRACE_2L_oam':
            calc = grace_fm("GRACE-2L-OAM")
        elif pot == 'GRACE_2L_omat':
            calc = grace_fm("GRACE-2L-OMAT")
        elif pot == 'GRACE_FS_oam':
            calc = grace_fm("GRACE-FS-OAM")
        elif pot == 'GRACE_FS_omat':
            calc = grace_fm("GRACE-FS-OMAT")
        elif pot == 'eqV2_31M_omat_mp_salex':
            current_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(current_dir, "eqV2_models", "eqV2_31M_omat_mp_salex.pt")
            calc = OCPCalculator(checkpoint_path=checkpoint_path, cpu=True, seed=123)
        elif pot == 'NEP_89':
            calc = NEP('nep89.txt')

    elif pot in pot_list_EIPs:
        list_EAM = ['FeMnNiCu_Bonny', 'FeTiCoNiCuMoW_Zhou', 'FeCrCoNiCu_Deluigi']
        list_MEAM = ['FeCu_Lee']
        if caltype == 'Fe_only':
            if pot in list_EAM:
                cmds = ["pair_style eam/alloy",
                        f"pair_coeff * * IPs/{pot}.eam.alloy Fe"]
            if pot in list_MEAM:
                if pot == 'FeCu_Lee':
                    cmds = ["pair_style meam",
                            f"pair_coeff * * IPs/library_FeCu_Lee.meam Fe Cu IPs/FeCu_Lee.meam Fe"]
            atom_types = {'Fe': 1}
        if caltype == 'sub':
            if pot in list_EAM:
                cmds = ["pair_style eam/alloy",
                        f"pair_coeff * * IPs/{pot}.eam.alloy Fe {sub_e_calc}"]
            if pot in list_MEAM:
                if pot == 'FeCu_Lee':
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


def get_e_list(pot):
    #e_list = ['Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo']
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_uMLIPs:
        e_list = ['Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo']
    if pot in pot_list_EIPs:
        if pot == 'FeMnNiCu_Bonny':
            e_list = ['Mn', 'Ni', 'Cu']
        if pot == 'FeTiCoNiCuMoW_Zhou':
            e_list = ['Ti', 'Co', 'Ni', 'Cu', 'Mo']
        if pot == 'FeCu_Lee':
            e_list = ['Cu']
        if pot == 'FeCrCoNiCu_Deluigi':
            e_list = ['Cr', 'Co', 'Ni', 'Cu']
    return e_list


########## Calculation Functions

def E_bulk_cal(pot, optimizer, fmax, steps): # Calculate the energy of pure bulk system.
    print(f"\n\n\n\n\n\n\n\nCalculating E_bulk....\nPotential: {pot}\n")
    pure_bulk = get_pure_bulk(pot)
    caltype = 'Fe_only'
    pure_bulk.calc = define_calc(pot, caltype)
    ucf = FrechetCellFilter(pure_bulk, hydrostatic_strain=True)
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    E_bulk = pure_bulk.get_potential_energy()
    n_steps = opt.nsteps

    dir_path = f"Results/{pot}/E_bulk"
    os.makedirs(dir_path, exist_ok=True)
    csv_path = os.path.join(dir_path, f"E_bulk_{pot}.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Potential', 'E_bulk (eV)', 'Fmax', 'n_steps', 'MaxSteps', 'Optimizer'])
        writer.writerow([pot, E_bulk, fmax, n_steps, steps, optimizer])

    print(f"\n\nE_bulk is recorded in {csv_path}.\n\n")


def E_bulk_sub_cal(pot, optimizer, fmax, steps): # Calculate the energy of bulk system with one solute.
    pure_bulk = get_pure_bulk(pot)
    e_list = get_e_list(pot)
    E_bulk_sub = np.zeros(len(e_list))

    for i in range(len(e_list)):
        sub_e = e_list[i]
        print(f"\n\n\n\n\n\n\n\nCalculating E_bulk_sub....\nPotential: {pot}   Element: {sub_e}\n")
        pure_bulk_sub = pure_bulk.copy()
        pure_bulk_sub[1].symbol = sub_e
        caltype = 'sub' #substituted
        pure_bulk_sub.calc = define_calc(pot, caltype, e_list[i])
        ucf = FrechetCellFilter(pure_bulk_sub, hydrostatic_strain=True)
        opt = set_optimizer(optimizer, ucf)
        opt.run(fmax=fmax, steps=steps)
        E_bulk_sub[i] = pure_bulk_sub.get_potential_energy()
        n_steps = opt.nsteps

        dir_path = f"Results/{pot}/E_bulk_sub"
        os.makedirs(dir_path, exist_ok=True)
        csv_path = os.path.join(dir_path, f"E_bulk_sub_{pot}_{sub_e}.csv")
        with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Potential', 'E_bulk_sub (eV)', 'Element', 'Fmax', 'n_steps', 'MaxSteps', 'Optimizer'])
            writer.writerow([pot, E_bulk_sub[i], sub_e, fmax, n_steps, steps, optimizer])

        print(f"\n\nE_bulk_sub is recorded in {csv_path}.\n\n")



def E_gb_cal(pot, gb, optimizer, fmax, steps): # Calculate the energy of pure GB system.
    print(f"\n\n\n\n\n\n\n\nCalculating E_gb....\nPotential: {pot}  GB: {gb}....\n")
    GB = get_GB(pot, gb)
    caltype = 'Fe_only'
    GB.calc = define_calc(pot, caltype)
    opt = set_optimizer(optimizer, GB)
    opt.run(fmax=fmax, steps=steps)
    E_gb = GB.get_potential_energy()
    n_steps = opt.nsteps

    dir_path = f"Results/{pot}/E_gb"
    os.makedirs(dir_path, exist_ok=True)
    csv_path = os.path.join(dir_path, f"E_gb_{pot}_{gb}.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Potential', 'E_gb (eV)', 'GB', 'Fmax', 'n_steps', 'MaxSteps', 'Optimizer'])
        writer.writerow([pot, E_gb, gb, fmax, n_steps, steps, optimizer])

    print(f"\n\nE_gb is recorded in {csv_path}.\n\n")


def E_gb_sub_cal(pot, gb, sub_e, optimizer, fmax, steps): # Calculate the energy of pure GB system with one solute.
    GB = get_GB(pot, gb)
    y_GB_plane = get_GB_plane(pot, gb) #get the coordinate of GB plane
    cal_sites = get_cal_sites(pot, gb)
    number_of_sites = len(cal_sites)
    E_gb_sub_all = []
    distances = []

    dir_path = f"Results/{pot}/E_gb_sub"
    os.makedirs(dir_path, exist_ok=True)
    csv_path = os.path.join(dir_path, f"E_gb_sub_{pot}_{gb}_{sub_e}.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Potential', 'E_gb_sub (eV)', 'GB', 'Site', 'Distance to GB (Angstrom)', 'Fmax', 'n_steps', 'MaxSteps', 'Optimizer'])

    for i in range(number_of_sites):
        GB_sub = GB.copy()
        GB_sub[cal_sites[i]].symbol = sub_e
        caltype = 'sub' #substituted
        GB_sub.calc = define_calc(pot, caltype, sub_e)
        opt = set_optimizer(optimizer, GB_sub)
        opt.run(fmax=fmax, steps=steps)
        E_gb_sub = GB_sub.get_potential_energy()
        n_steps = opt.nsteps
        E_gb_sub_all.append(E_gb_sub)
        sub_atom = GB_sub[cal_sites[i]]
        distance = abs(sub_atom.position[1] - y_GB_plane) #distance from site to GB plane
        tolerance = 0.1 # Tolerance that defining if the site is on the GB plane.
        if distance < tolerance:
            distance = 0
            distances.append(distance)
        else:
            distances.append(distance)

        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([pot, E_gb_sub, gb, cal_sites[i], distance, fmax, n_steps, steps, optimizer])

        print(f"\n\nE_gb_sub is recorded in {csv_path}.\n\n")


def pre_relax_GB(pot, gb, optimizer, fmax, steps):
    print(f"\n\nRelaxing {pot}, {gb}....\n")
    GB = get_GB_rescaled(pot, gb)
    caltype = 'Fe_only'
    GB.calc = define_calc(pot, caltype)
    opt = set_optimizer(optimizer, GB)
    opt.run(fmax=fmax, steps=steps)
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_uMLIPs:
        write(f'Structures/GBs/Relaxed/{pot}_{gb}_relaxed.vasp', GB)
    if pot in pot_list_EIPs:
        write(f'Structures/GBs/Relaxed/{pot}_{gb}_relaxed.lammps-data', GB)


def pre_relax_GB_novac(pot, gb, optimizer, fmax, steps):
    print(f"\n\nRelaxing {pot}, {gb}....\n")
    GB = get_GB_novac_unrelaxed(pot, gb)
    caltype = 'Fe_only'
    GB.calc = define_calc(pot, caltype)
    ucf = FrechetCellFilter(GB,
                            mask = [0, 0, 1, 0, 0, 0], #(xx, yy, zz, yz, xz, xy) True: relax to 0, False: fix
                            hydrostatic_strain=True)
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_uMLIPs:
        write(f'Structures/GBs/No_vac_relaxed/{pot}_{gb}_relaxed.vasp', GB)
    if pot in pot_list_EIPs:
        write(f'Structures/GBs/No_vac_relaxed/{pot}_{gb}_relaxed.lammps-data', GB)