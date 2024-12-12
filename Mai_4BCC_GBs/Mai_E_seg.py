from ase.constraints import UnitCellFilter
import numpy as np
import sys
from ase.io.trajectory import Trajectory
from Mai_ASE_setup import get_pure_bulk, get_GB, get_e_list, get_lattice_parameter, get_cal_sites, get_bulk_number_of_atom, get_GB_number_of_atom, get_z_GB_plane
from Mai_ASE_setup import define_calc, set_optimizer

######################################################################################################################
                                    # Functions for calculations
######################################################################################################################


def bulk_energy_cal(potchoice, optimizer, fmax, steps): # Calculate the energy of pure bulk system.
    print("\n\nCalculating E_bulk....\n")
    pure_bulk = get_pure_bulk(potchoice)
    caltype = 'Fe_only'
    pure_bulk.calc = define_calc(potchoice, caltype)
    ucf = UnitCellFilter(pure_bulk,
                         hydrostatic_strain=True,
                         cell_factor=float(len(pure_bulk)*10))
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    bulk_energy_after_relax = pure_bulk.get_potential_energy()
    with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "a") as file:
        file.write(f"\n\n----- Energies of the Bulk System -----\n\nThe energy of the bulk system is {bulk_energy_after_relax} eV.\n")
    print(f"\n\nThe energy of the Fe bulk system is {bulk_energy_after_relax} eV.\nPotential: {potchoice}    GB: {GBchoice}\n\n")
    return bulk_energy_after_relax


def bulk_sub_energy_cal(potchoice, optimizer, fmax, steps): # Calculate the energy of bulk system with one solute.
    pure_bulk = get_pure_bulk(potchoice)
    print("\n\n\n\n\n\n\n\nCalculating E_bulk_sub....\n")
    e_list = get_e_list(potchoice)
    bulk_energy_after_opt = np.zeros(len(e_list))
    for i in range(len(e_list)):
        pure_bulk_sub = pure_bulk.copy()
        pure_bulk_sub[1].symbol = e_list[i]
        caltype = 'sub' #substituted
        pure_bulk_sub.calc = define_calc(potchoice, caltype, e_list[i])
        ucf = UnitCellFilter(pure_bulk_sub,
                             hydrostatic_strain=True,
                             cell_factor=float(len(pure_bulk_sub)*10))
        opt = set_optimizer(optimizer, ucf)
        opt.run(fmax=fmax, steps=steps)
        bulk_energy_after_opt[i] = pure_bulk_sub.get_potential_energy()
        with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "a") as file:
            file.write(f"\nThe energy of the bulk system substituded by one {e_list[i]} atom is {bulk_energy_after_opt[i]} eV.")
        print(f"\n\nThe energy of the bulk system substituded by one {e_list[i]} is {bulk_energy_after_opt[i]} eV.\nPotential: {potchoice}    GB: {GBchoice}\n\n")
    return bulk_energy_after_opt


def gb_energy_cal(potchoice, GBchoice, optimizer, fmax, steps): # Calculate the energy of pure GB system.
    print("\n\n\n\n\n\n\n\nCalculating E_gb....\n")
    GB = get_GB(potchoice,GBchoice)
    caltype = 'Fe_only'
    GB.calc = define_calc(potchoice, caltype)
    opt = set_optimizer(optimizer, GB)
    opt.run(fmax=fmax, steps=steps)
    E_gb_after_opt = GB.get_potential_energy()
    with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "a") as file:
        file.write(f"\n\n----- Energies of GB System -----\n\nThe energy of the GB system is {E_gb_after_opt} eV.")
    print(f"\n\nThe energy of the GB system is {E_gb_after_opt} eV.\nPotential: {potchoice}    GB: {GBchoice}\n")
    return E_gb_after_opt


def gb_sub_energy_cal(potchoice, GBchoice, cal_sites, sub_e, z_GB_plane, optimizer, fmax, steps, savetraj): # Calculate the energy of pure GB system with one solute.
    GB = get_GB(potchoice, GBchoice)
    print("\n\n\n\n\n\n\n\nCalculating E_gb_sub of {}....\n".format(sub_e))
    with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "a") as file:
        file.write(f"\n\n-- Substitution of {sub_e} --")
    number_of_sites = len(cal_sites)
    GB_sub_energys = []
    distances = []
    for i in range(number_of_sites):
        GB_sub = GB.copy()
        GB_sub[cal_sites[i]].symbol = sub_e
        caltype = 'sub' #substituted
        GB_sub.calc = define_calc(potchoice, caltype, sub_e)
        if savetraj == 'Yes':
            traj = Trajectory(f'Results/Trajectories/GB_{potchoice}_{GBchoice}_{sub_e}.traj', 'w', GB_sub)
        opt = set_optimizer(optimizer, GB_sub)
        if savetraj == 'Yes':
            opt.attach(traj.write, interval=10)
        opt.run(fmax=fmax, steps=steps)
        e = GB_sub.get_potential_energy()
        GB_sub_energys.append(e)
        atom = GB_sub[cal_sites[i]]
        d = abs(atom.position[2] - z_GB_plane)
        tolerance = 0.1 # Tolerance that defining if the site is on the GB plane.
        if d < tolerance:
            d = 0
            distances.append(d)
        else:
            distances.append(d)
        with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "a") as file:
            file.write(f"\nThe E_gb_sub with site {cal_sites[i]} (distance from GB plane: {distances[i]} angstrom) substituded by one {sub_e} atom at site {cal_sites[i]} is {GB_sub_energys[i]} eV.")
        print(f"\n\nThe E_gb_sub with site {cal_sites[i]}  (distance from GB plane: {distances[i]} angstrom) substituded by one {sub_e} atom at site {cal_sites[i]} is {GB_sub_energys[i]} eV.\nPotential: {potchoice}    GB: {GBchoice}\n")
    return GB_sub_energys, distances


def gb_sub_energy_cal_4alle(potchoice, GBchoice, cal_sites, z_GB_plane, optimizer, fmax, steps, savetraj): # Make the function gb_sub_energy_cal applicable to a list of elements.
    #Calculate the lowest GB segregation energy
    E_gb_sub_ls4alle = []
    distances_ls4alle = []
    e_list = get_e_list(potchoice)
    for i in range(len(e_list)):
        sub_e = e_list[i]
        GB_sub_energys, distances = gb_sub_energy_cal(potchoice, GBchoice, cal_sites, sub_e, z_GB_plane, optimizer, fmax, steps, savetraj)
        E_gb_sub_ls4alle.append(GB_sub_energys)
        distances_ls4alle.append(distances)
    return E_gb_sub_ls4alle, distances_ls4alle # E_gb_sub_ls4alle[[a list of energy of system with one substitutional solute at different sites] different solute elements].


def segregation_energy_cal(potchoice, GBchoice, optimizer, fmax, steps, savetraj='Yes'):
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
    z_GB_plane = get_z_GB_plane(potchoice, GBchoice)
    cal_sites = get_cal_sites(GBchoice)
    bulk_number_of_atom = get_bulk_number_of_atom(potchoice)
    GB_number_of_atom = get_GB_number_of_atom(potchoice, GBchoice)
    e_list = get_e_list(potchoice)
    with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "w") as file:
        file.write(
            f"Potential: {potchoice}\nLattice parameter: {a0_Fe}\nGB: {GBchoice}\nAtom number GB: {GB_number_of_atom}\nAtom number bulk: {bulk_number_of_atom}\nSolute elements: {e_list}\nIndices of calculated sites: {cal_sites}\nOptimizer: {optimizer}\nFmax: {fmax}\n")
    E_bulk = bulk_energy_cal(potchoice, optimizer, fmax, steps)
    E_bulk_sub = bulk_sub_energy_cal(potchoice, optimizer, fmax, steps)
    E_gb = gb_energy_cal(potchoice, GBchoice, optimizer, fmax, steps)
    E_gb_sub, site_dist = gb_sub_energy_cal_4alle(potchoice, GBchoice, cal_sites, z_GB_plane, optimizer, fmax, steps, savetraj)
    E_seg = np.zeros((len(e_list), len(cal_sites)), dtype=int).tolist()
    with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "a") as file:
        file.write(f"\n\n----- Segregation Energies -----")
    for e in range(len(e_list)):
        with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "a") as file:
            file.write(f"\n\n-- Segregation energy of {e_list[e]} --")
        for s in range(len(cal_sites)):
            E_seg[e][s] = (E_gb_sub[e][s] - E_gb) - (E_bulk_sub[e] - E_bulk)
            with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "a") as file:
                file.write(f"\nThe segregation energy of one {e_list[e]} atom at site {cal_sites[s]} (distance from GB plane: {site_dist[e][s]} angstrom) in Fe GB is {E_seg[e][s]} eV.")
    # Find the index of the minimum segregation energy for each element.
    E_seg_min = []
    E_seg_min_index = []
    with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "a") as file:
        file.write(f"\n\n-- Lowest segregation energy --")
    for i in range(len(e_list)):
        E_seg_min.append(min(E_seg[i]))
        index = E_seg[i].index(E_seg_min[i])
        E_seg_min_index.append(index)
        with open(f"Results/Details/{potchoice}/{GBchoice}.txt", "a") as file:
            file.write(f"\nThe lowest segregation energy of one {e_list[i]} atom in Fe GB is {E_seg_min[i]} eV at site {cal_sites[E_seg_min_index[i]]}.")
        print(f"The lowest segregation energy of one {e_list[i]} atom in Fe GB is {E_seg_min[i]} eV at site {cal_sites[E_seg_min_index[i]]}.\nPotential: {potchoice}    GB: {GBchoice}\n")
    return E_seg_min


######################################################################################################################
                                        # Execution
######################################################################################################################
''' If divide the jobs for each potential and GB

if __name__ == "__main__":
    potchoice = sys.argv[1]
    GBchoice = sys.argv[2]
    steps = 5000
    fmax = 1e-4  # Global fmax
    optimizer = 'FIRE'  # Global optmizer
    E_seg_min = segregation_energy_cal(potchoice, GBchoice, optimizer, fmax, steps, savetraj='Yes')
    with open(f"Results/E_seg_min/{potchoice}_{GBchoice}.txt", "w") as file:
        file.write(f"E_seg_min_data_{potchoice}_{GBchoice} = {E_seg_min}\n")
'''


'''
pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']
create_needed_dir(pot_list)
fmax = 1e-4 # Global fmax
steps = 3000
optimizer = 'BFGS' # Global optmizer

with open(f"Results/4Data_analysis.txt", "w") as file:
    file.write("Used for lowest segregation energies comparison.\n")
for potchoice in pot_list:
    E_seg_pot = []
    for GBchoice in GB_list:
        E_seg_min = segregation_energy_cal(potchoice, GBchoice, optimizer,  fmax, steps)
        E_seg_pot.append(E_seg_min)
    with open(f"Results/4Data_analysis.txt", "a") as file:
        file.write(f"{potchoice}_data = {E_seg_pot}\n")
'''

