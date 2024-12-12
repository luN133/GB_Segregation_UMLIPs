from ase.io import read
from ase.constraints import UnitCellFilter
from Marcel_ASE_setup import set_optimizer, define_calc



######################################################################################################################
                                    # Functions for calculations
######################################################################################################################

def cal_H_Fe12C4(potchoice, fmax, optimizer, steps):
    Fe12C4 = read('Structures/FM_cementite.vasp')
    Fe12C4.calc = define_calc(potchoice)
    ucf = UnitCellFilter(Fe12C4, cell_factor=10*len(Fe12C4), hydrostatic_strain=True)
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    H_Fe12C4 = Fe12C4.get_potential_energy()
    print(f'Potential: {potchoice}      Un_sub cementite')
    return H_Fe12C4


def cal_H_cementite_8d(potchoice, solute, fmax, optimizer, steps):
    Fe12C4 = read('Structures/FM_cementite.vasp')
    Fe11MC4_8d = Fe12C4.copy()
    Fe11MC4_8d[5].symbol = solute
    caltype = 'sub'
    Fe11MC4_8d.calc = define_calc(potchoice, caltype)
    ucf = UnitCellFilter(Fe11MC4_8d, cell_factor=10*len(Fe11MC4_8d), hydrostatic_strain=True)
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    H_Fe11MC4_8d = Fe11MC4_8d.get_potential_energy()
    print(f'Potential: {potchoice}      Solute: {solute}        Current site: 8d')
    return H_Fe11MC4_8d


def cal_H_cementite_4c(potchoice, solute, fmax, optimizer, steps):
    Fe12C4 = read('Structures/FM_cementite.vasp')
    Fe11MC4_4c = Fe12C4.copy()
    Fe11MC4_4c[9].symbol = solute
    caltype = 'sub'
    Fe11MC4_4c.calc = define_calc(potchoice, caltype)
    ucf = UnitCellFilter(Fe11MC4_4c, cell_factor=10*len(Fe11MC4_4c), hydrostatic_strain=True)
    opt = set_optimizer(optimizer, ucf)
    opt.run(fmax=fmax, steps=steps)
    H_Fe11MC4_4c = Fe11MC4_4c.get_potential_energy()
    print(f'Potential: {potchoice}      Solute: {solute}        Current site: 4c')
    return H_Fe11MC4_4c


def cal_deltaH_Fe11MC4_4allpotalle(pot_list, e_list, fmax, optimizer, steps):
    deltaH_list_all_8d = []
    deltaH_list_all_4c = []
    for potchoice in pot_list:
        deltaH_list_pot_8d = []
        deltaH_list_pot_4c = []
        H_Fe12C4 = cal_H_Fe12C4(potchoice, fmax, optimizer, steps)
        for solute in e_list:
            H_Fe11MC4_8d = cal_H_cementite_8d(potchoice, solute, fmax, optimizer, steps)
            H_Fe11MC4_4c = cal_H_cementite_4c(potchoice, solute, fmax, optimizer, steps)
            deltaH_Fe11MC4_8d = H_Fe11MC4_8d - H_Fe12C4
            deltaH_Fe11MC4_4c = H_Fe11MC4_4c - H_Fe12C4
            deltaH_Fe11MC4_8d = round(deltaH_Fe11MC4_8d, 4)
            deltaH_Fe11MC4_8d = deltaH_Fe11MC4_8d / 16 #atom number 16 unit eV/atom
            deltaH_Fe11MC4_4c = deltaH_Fe11MC4_4c / 16
            deltaH_Fe11MC4_4c = round(deltaH_Fe11MC4_4c, 4)
            deltaH_list_pot_8d.append(deltaH_Fe11MC4_8d)
            deltaH_list_pot_4c.append(deltaH_Fe11MC4_4c)
        deltaH_list_all_8d.append(deltaH_list_pot_8d)
        deltaH_list_all_4c.append(deltaH_list_pot_4c)
    return deltaH_list_all_8d, deltaH_list_all_4c

######################################################################################################################
                                        # Execution
######################################################################################################################


pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
e_list = ['Al', 'Si', 'P', 'S', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Nb', 'Mo', 'W']
steps = 500
fmax = 1e-4
optimizer = 'BFGS'
deltaH_results_8d, deltaH_results_4c = cal_deltaH_Fe11MC4_4allpotalle(pot_list, e_list, fmax, optimizer, steps)
with open("deltaH_Fe11MC4_Results.txt", 'w') as file:
    file.write(f"Potentials: {pot_list}\nAlloying elements: {e_list}\n\n")
for i in range(len(pot_list)):
    with open("deltaH_Fe11MC4_Results.txt", 'a') as file:
        file.write(f"\ndata_{pot_list[i]}_8d = {deltaH_results_8d[i]}\ndata_{pot_list[i]}_4c = {deltaH_results_4c[i]}")
for i1 in range(len(pot_list)):
    with open("deltaH_Fe11MC4_Results.txt", 'a') as file:
        file.write(f"\n\n\n----- Solution Enthalpy {pot_list[i1]} -----\n")
    for i2 in range(len(e_list)):
        with open("deltaH_Fe11MC4_Results.txt", 'a') as file:
            file.write(f"\n\n{e_list[i2]}: site 8d: {deltaH_results_8d[i1][i2]} eV        site 4c: {deltaH_results_4c[i1][i2]} eV")