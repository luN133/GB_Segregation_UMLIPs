from ase.io import read
from chgnet.model import CHGNetCalculator
from mace.calculators import mace_mp
from ase.constraints import UnitCellFilter
from ase.optimize import BFGS, FIRE
from chgnet.model.model import CHGNet
import matgl
from matgl.ext.ase import M3GNetCalculator
import torch
import sys
import os
from ase.io.trajectory import Trajectory
from ase.calculators.lammpslib import LAMMPSlib
from sevenn.sevennet_calculator import SevenNetCalculator


######################################################################################################################
                                    # Functions for simplicity
######################################################################################################################

# This function is only for convenience, and making sure that in the output the name of the model is correct.

def get_pure_bulk(potchoice):
    pot_list_uMLIPs = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
    pot_list_EIPs = ['FeMnNiCu_Bonny', 'FeCrMnNi_Daramola', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
    if potchoice in pot_list_EIPs:
        pure_bulk = read(f"Structures/Pure_bulks/pure_bulk_{potchoice}.lammps")
        pure_bulk.symbols = 'Fe'
    if potchoice in pot_list_uMLIPs:
        pure_bulk = read(f"Structures/Pure_bulks/pure_bulk_{potchoice}.vasp")
    return pure_bulk


def get_GB(potchoice, GBchoice):
    pot_list_uMLIPs = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
    pot_list_EIPs = ['FeMnNiCu_Bonny', 'FeCrMnNi_Daramola', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
    if potchoice in pot_list_EIPs:
        GB = read(f"Structures/GBs/Relaxed/{potchoice}_{GBchoice}_relaxed.lammps-data")
        GB.symbols = 'Fe'
    if potchoice in pot_list_uMLIPs:
        GB = read(f"Structures/GBs/Relaxed/{potchoice}_{GBchoice}_relaxed.vasp")
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
    elif potchoice == 'FeCrMnNi_Daramola':
        a0_Fe = 3.498696161797819
    elif potchoice == 'FeTiCoNiCuMoW_Zhou':
        a0_Fe = 3.6284420233011447
    elif potchoice == 'FeCu_Lee':
        a0_Fe = 3.611098367149702
    return a0_Fe


def define_cal(potchoice, caltype=None, sub_e_calc=None):
    pot_list_uMLIPs = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
    pot_list_EIPs = ['FeMnNiCu_Bonny', 'FeCrMnNi_Daramola', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
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
        list_EAM = ['FeMnNiCu_Bonny', 'FeCrMnNi_Daramola', 'FeTiCoNiCuMoW_Zhou']
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


######################################################################################################################
                                    # Functions for calculations
######################################################################################################################

def bulk_energy_cal(potchoice, GBchoice, sub_e, optimizer, fmax, steps): # Calculate the energy of pure bulk system.
    print("\n\nCalculating E_bulk....\n")
    pure_bulk = get_pure_bulk(potchoice)
    caltype = 'Fe_only'
    pure_bulk.calc = define_cal(potchoice, caltype)
    #relaxing the bulk crystal
    ucf = UnitCellFilter(pure_bulk,
                         hydrostatic_strain=True)
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    bulk_energy_after_relax = pure_bulk.get_potential_energy()
    with open(f"Results/Details/{potchoice}/{GBchoice}/{sub_e}.txt", "a") as file:
        file.write(f"\n\n----- Energies of the Bulk System -----\n\nThe energy of the bulk system is {bulk_energy_after_relax} eV.\n")
    print(f"\n\nThe energy of the Fe bulk system is {bulk_energy_after_relax} eV.\nPotential: {potchoice}    GB: {GBchoice}\n\n")
    return bulk_energy_after_relax


def bulk_sub_energy_cal(potchoice, GBchoice, sub_e, optimizer, fmax, steps): # Calculate the energy of bulk system with one solute.
    print("\n\n\n\n\n\n\n\nCalculating E_bulk_sub....\n")
    pure_bulk = get_pure_bulk(potchoice)
    pure_bulk_sub = pure_bulk.copy()
    pure_bulk_sub[0].symbol = sub_e
    caltype = 'sub'
    pure_bulk_sub.calc = define_cal(potchoice, caltype, sub_e)
    ucf = UnitCellFilter(pure_bulk_sub, hydrostatic_strain=True, cell_factor=float(len(pure_bulk_sub)*10))
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    bulk_energy_after_opt = pure_bulk_sub.get_potential_energy()
    with open(f"Results/Details/{potchoice}/{GBchoice}/{sub_e}.txt", "a") as file:
        file.write(f"\nThe energy of the bulk system substituded by one {sub_e} atom is {bulk_energy_after_opt} eV.")
    print(f"\n\nThe energy of the bulk system substituded by one {sub_e} is {bulk_energy_after_opt} eV.\nPotential: {potchoice}    GB: {GBchoice}\n\n")
    return bulk_energy_after_opt


def gb_energy_cal(potchoice, GBchoice, sub_e, optimizer, fmax, steps): # Calculate the energy of pure GB system.
    print("\n\n\n\n\n\n\n\nCalculating E_gb....\n")
    GB = get_GB(potchoice, GBchoice)
    caltype = 'Fe_only'
    GB.calc = define_cal(potchoice, caltype)
    opt = set_optimizer(optimizer, GB)
    opt.run(fmax=fmax, steps=steps)
    E_gb_after_opt = GB.get_potential_energy()
    with open(f"Results/Details/{potchoice}/{GBchoice}/{sub_e}.txt", "a") as file:
        file.write(f"\n\n----- Energies of GB System -----\n\nThe energy of the GB system is {E_gb_after_opt} eV.")
    print(f"\n\nThe energy of the GB system is {E_gb_after_opt} eV.\nPotential: {potchoice}    GB: {GBchoice}\n")
    return E_gb_after_opt


def gb_sub_energy_cal(potchoice, GBchoice, sub_e, cal_sites, y_GB_plane, optimizer, fmax, steps, savetraj): # Calculate the energy of pure GB system with one solute.
    print("\n\n\n\n\n\n\n\nCalculating E_gb_sub of {}....\n".format(sub_e))
    with open(f"Results/Details/{potchoice}/{GBchoice}/{sub_e}.txt", "a") as file:
        file.write(f"\n\n-- Substitution of {sub_e} --")
    GB = get_GB(potchoice, GBchoice)
    number_of_sites = len(cal_sites)
    GB_sub_energys = []
    distances = []
    for i in range(number_of_sites):
        GB_sub = GB.copy()
        GB_sub[cal_sites[i]].symbol = sub_e
        caltype = 'sub'
        GB_sub.calc = define_cal(potchoice, caltype, sub_e)
        if savetraj == 'Yes':
            traj = Trajectory(f'Results/Trajectories/GB_{potchoice}_{GBchoice}_{sub_e}.traj', 'w', GB_sub)
        opt = set_optimizer(optimizer, GB_sub)
        if savetraj == 'Yes':
            opt.attach(traj.write, interval=10)
        opt.run(fmax=fmax, steps=steps)
        e = GB_sub.get_potential_energy()
        GB_sub_energys.append(e)
        atom = GB_sub[cal_sites[i]]
        d = abs(atom.position[1] - y_GB_plane)
        tolerance = 0.1 # Tolerance that defining if the site is on the GB plane.
        if d < tolerance:
            d = 0
            distances.append(d)
        else:
            distances.append(d)
        with open(f"Results/Details/{potchoice}/{GBchoice}/{sub_e}.txt", "a") as file:
            file.write(f"\nThe E_gb_sub with site {cal_sites[i]} (distance from GB plane: {distances[i]} angstrom) substituded by {sub_e} atom is {GB_sub_energys[i]} eV.")
        print(f"\n\nThe E_gb_sub with site {cal_sites[i]}  (distance from GB plane: {distances[i]} angstrom) substituded by {sub_e} atom is {GB_sub_energys[i]} eV.\nPotential: {potchoice}    GB: {GBchoice}\n")
    return GB_sub_energys, distances


def segregation_energy_cal(potchoice, GBchoice, sub_e, optimizer, fmax, steps, savetraj):
    '''
    Segregation energy:
    E_seg = E_gb_(n-1)_Fe,X - E_gb - E_bulk(m-1)Fe,X -E_bulk
    E_seg: Segregation Energy
    E_gb_(n-1)_Fe,X: the total energy of a grain boundary (GB) structure with n atoms, one of which is the substitutional solute X
    E_gb: the total energy of a grain boundary (GB) structure with n atoms
    E_bulk(m-1)Fe,X: the total energy of a bulk cell with m atoms, one of which is the substitutional solute X
    E_bulk: the energy of the bulk cell in its pure form, containing m atoms of iron (Fe)
    '''
    a0_Fe = get_lattice_parameter(potchoice)
    y_GB_plane = 15
    cal_sites = get_cal_sites(potchoice, GBchoice)
    bulk_number_of_atom = get_bulk_number_of_atom(potchoice)
    GB_number_of_atom = get_GB_number_of_atom(potchoice, GBchoice)
    with open(f"Results/Details/{potchoice}/{GBchoice}/{sub_e}.txt", "w") as file:
        file.write(
            f"Potential: {potchoice}\nLattice parameter: {a0_Fe}\nGB: {GBchoice}\nAtom number GB: {GB_number_of_atom}\nAtom number bulk: {bulk_number_of_atom}\nSolute element: {sub_e}\nIndices of calculated sites: {cal_sites}\nOptimizer: {optimizer}\nFmax: {fmax}\n")
    E_bulk = bulk_energy_cal(potchoice, GBchoice, sub_e, optimizer, fmax, steps)
    E_bulk_sub = bulk_sub_energy_cal(potchoice, GBchoice, sub_e, optimizer, fmax, steps)
    E_gb = gb_energy_cal(potchoice, GBchoice, sub_e, optimizer, fmax, steps)
    E_gb_sub, site_dist = gb_sub_energy_cal(potchoice, GBchoice, sub_e, cal_sites, y_GB_plane, optimizer, fmax, steps, savetraj)
    E_seg = []
    with open(f"Results/Details/{potchoice}/{GBchoice}/{sub_e}.txt", "a") as file:
        file.write(f"\n\n----- Segregation Energies -----")
    for s in range(len(cal_sites)):
        E_seg_i = (E_gb_sub[s] - E_gb) - (E_bulk_sub - E_bulk)
        E_seg.append(E_seg_i)
        with open(f"Results/Details/{potchoice}/{GBchoice}/{sub_e}.txt", "a") as file:
            file.write(f"\nThe segregation energy of one {sub_e} atom at site {cal_sites[s]} (distance from GB plane: {site_dist[s]} angstrom) in Fe GB is {E_seg[s]} eV.")
    # Find the index of the minimum segregation energy for each element.
    with open(f"Results/Details/{potchoice}/{GBchoice}/{sub_e}.txt", "a") as file:
        file.write(f"\n\n-- Lowest segregation energy --")
    E_seg_min = min(E_seg)
    index = E_seg.index(E_seg_min)
    with open(f"Results/Details/{potchoice}/{GBchoice}/{sub_e}.txt", "a") as file:
        file.write(f"\nThe lowest segregation energy of one {sub_e} atom in Fe GB is {E_seg_min} eV at site {cal_sites[index]}.")
    print(f"The lowest segregation energy of one {sub_e} atom in Fe GB is {E_seg_min} eV at site {cal_sites[index]}.\nPotential: {potchoice}    GB: {GBchoice}\n")
    return E_seg_min


######################################################################################################################
                                        # Execution
######################################################################################################################

if __name__ == "__main__":
    potchoice = sys.argv[1]
    GBchoice = sys.argv[2]
    sub_e = sys.argv[3]
    if potchoice == 'SevenNet':
        steps = 2000
    elif potchoice == 'M3GNet':
        steps = 5000
    else:
        steps = 1000
    fmax = 1e-4  # Global fmax
    optimizer = 'BFGS'  # Global optmizer
    E_seg_min = segregation_energy_cal(potchoice, GBchoice, sub_e, optimizer, fmax, steps, savetraj='Yes')
    with open(f"Results/E_seg_min/{potchoice}_{GBchoice}_{sub_e}.txt", "w") as file:
        file.write(f"E_seg_min_data_{potchoice}_{GBchoice}_{sub_e} = {E_seg_min}\n")
