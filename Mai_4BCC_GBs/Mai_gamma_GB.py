
from ase.filters import UnitCellFilter
import os
from Mai_ASE_setup import get_pure_bulk, define_calc, set_optimizer, get_GB_novac_relaxed, get_bulk_number_of_atom


######################################################################################################################
                                    # Functions for calculations
######################################################################################################################

def bulk_energy_cal(potchoice, fmax, optimizer, steps): # Calculate the energy of pure bulk system.
    print(f"\n\n\n\nCalculating E_bulk....     Potential: {potchoice}\n")
    pure_bulk = get_pure_bulk(potchoice)
    caltype = 'Fe_only'
    pure_bulk.calc = define_calc(potchoice, caltype)
    #relaxing the bulk crystal
    ucf = UnitCellFilter(pure_bulk,
                         hydrostatic_strain=True,
                         cell_factor=float(len(pure_bulk)*10))
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    bulk_energy_after_relax = pure_bulk.get_potential_energy()
    return bulk_energy_after_relax


def gb_energy_cal(potchoice, GBchoice, fmax, optimizer, steps): # Calculate the energy of pure GB system.
    print(f"\n\n\n\nCalculating E_gb....     Potential: {potchoice}    GB: {GBchoice}\n")
    GB = get_GB_novac_relaxed(potchoice, GBchoice)
    caltype = 'Fe_only'
    GB.calc = define_calc(potchoice, caltype)
    opt = set_optimizer(optimizer, GB)
    opt.run(fmax=fmax, steps=steps)
    E_gb_after_opt = GB.get_potential_energy()
    return E_gb_after_opt


'''
Grain boundary energy:
Gamma_gb = (E_gb - N_gb/N_bulk * E_bulk) / (2*A)
N_gb, N_bulk: number of atom in the GB cell and bulk cell, respectively
A: the interfacial area present in the GB cell.
The factor of two in the denominator is due to the periodic condition resulting in two GB interfaces present at the middle and top/bottom of the cell.
'''


def gamma_GB_cal(potchoice, GBchoice, fmax, optimizer, steps):
    GB = get_GB_novac_relaxed(potchoice, GBchoice)
    bulk_number_of_atom = get_bulk_number_of_atom(potchoice)
    GB_number_of_atom = GB.get_global_number_of_atoms()
    A = (GB.cell[0][0] * GB.cell[1][1])  # in angstrom^2
    E_bulk = bulk_energy_cal(potchoice, fmax, optimizer, steps)
    E_gb = gb_energy_cal(potchoice, GBchoice, fmax, optimizer, steps)
    Gamma_gb = (E_gb - (GB_number_of_atom / bulk_number_of_atom) * E_bulk) / (2 * A)  # eV/angstrom^2
    Gamma_gb_unit_converted = Gamma_gb * 1.602176565e-19 / 1e-20  # J/m^2
    return Gamma_gb_unit_converted


def run_cal (pot_list, GB_list, fmax, optimizer, steps):
    Gamma_GB = []
    os.makedirs('Results', exist_ok=True)
    with open(f"Results/gamma_GB.txt", "w") as file:
        file.write(f"Potentials: {pot_list}\nGBs: {GB_list}\nOptimizer: {optimizer}\nFmax: {fmax}\nMaxsteps: {steps}\n")
    for potchoice in pot_list:
        Gamma_GB_per_pot = []
        for GBchoice in GB_list:
            gamma_GB = gamma_GB_cal(potchoice, GBchoice, fmax, optimizer, steps)
            Gamma_GB_per_pot.append(gamma_GB)
        Gamma_GB.append(Gamma_GB_per_pot)
    with open(f"Results/gamma_GB.txt", "a") as file:
        file.write(f"gamma_GB_data = {Gamma_GB}")


pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']
steps = 1000
optimizer = 'FIRE'
fmax = 1e-4
run_cal(pot_list, GB_list, fmax, optimizer, steps)