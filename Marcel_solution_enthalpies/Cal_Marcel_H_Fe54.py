from ase.constraints import UnitCellFilter
from Marcel_ASE_setup import build_Fe54, define_calc, set_optimizer


######################################################################################################################
                                    # Functions for calculations
######################################################################################################################

def cal_H_Fe54(potchoice, fmax, optimizer):
    Fe54 = build_Fe54(potchoice)
    caltype = 'Fe_only'
    Fe54.calc = define_calc(potchoice, caltype)
    ucf = UnitCellFilter(Fe54, cell_factor=10*len(Fe54), hydrostatic_strain=True)
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax)
    H_Fe54 = Fe54.get_potential_energy()
    return H_Fe54


def cal_H_Fe53M(potchoice, solute, fmax, optimizer):
    Fe54 = build_Fe54(potchoice)
    Fe53M = Fe54.copy()
    Fe53M[0].symbol = solute
    sub_e = 'Cu'
    caltype = 'sub'
    Fe53M.calc = define_calc(potchoice, caltype, sub_e)
    ucf = UnitCellFilter(Fe53M, cell_factor=10 * len(Fe54), hydrostatic_strain=True)
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax)
    H_Fe53M = Fe53M.get_potential_energy()
    return H_Fe53M


def cal_deltaH_Fe53M_4allpotalle(pot_list, e_list, fmax=1e-4, optimizer='BFGS'):
    deltaH_list_all = []
    for potchoice in pot_list:
        deltaH_list_pot = []
        H_Fe54 = cal_H_Fe54(potchoice, fmax, optimizer)
        for solute in e_list:
            H_Fe53M = cal_H_Fe53M(potchoice, solute, fmax, optimizer)
            deltaH_Fe53M = H_Fe53M - H_Fe54
            deltaH_Fe53M = round(deltaH_Fe53M, 4)
            deltaH_list_pot.append(deltaH_Fe53M)
        deltaH_list_all.append(deltaH_list_pot)
    return deltaH_list_all


######################################################################################################################
                                        # Execution
######################################################################################################################

pot_list = ['FeMnNiCu_Bonny', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
#e_list = ['Al', 'Si', 'P', 'S', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Nb', 'Mo', 'W']
e_list = ['Cu']
fmax = 1e-4
optimizer = 'FIRE'
deltaH_results = cal_deltaH_Fe53M_4allpotalle(pot_list, e_list, fmax, optimizer)
with open("deltaH_Fe54_Results.txt", 'w') as file:
    file.write(f"Potentials: {pot_list}\nAlloying elements: {e_list}\n\n")
for i in range(len(pot_list)):
    with open("deltaH_Fe54_Results.txt", 'a') as file:
        file.write(f"\ndata_{pot_list[i]} = {deltaH_results[i]}")
for i1 in range(len(pot_list)):
    with open("deltaH_Fe54_Results.txt", 'a') as file:
        file.write(f"\n\n\n----- Solution Enthalpy {pot_list[i1]} -----\n")
    for i2 in range(len(e_list)):
        with open("deltaH_Fe54_Results.txt", 'a') as file:
            file.write(f"\n{e_list[i2]}: {deltaH_results[i1][i2]} eV")



