from ase.io import write, read
from FCC.Functions_FCC import define_calc, set_optimizer, get_pot_list_EIPs,get_pot_list_uMLIPs, get_GB_novac_unrelaxed
from ase.filters import FrechetCellFilter


def get_GB(pot, gb):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_EIPs:
        GB = read(f"GBs/Relaxed/{pot}_{gb}_relaxed.lammps-data")
        GB.symbols = 'Fe'
    elif pot in pot_list_uMLIPs:
        GB = read(f"GBs/Relaxed/{pot}_{gb}_relaxed.vasp")
    return GB


def get_GB_rescaled(pot, gb):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_EIPs:
        GB = read(f"GBs/Rescaled/{pot}_{gb}.lammps-data")
        GB.symbols = 'Fe'
    elif pot in pot_list_uMLIPs:
        GB = read(f"GBs/Rescaled/{pot}_{gb}.vasp")
    return GB


def relax_GB(pot, gb, optimizer, fmax, steps):
    print(f"\n\nRelaxing {pot}, {gb}....\n")
    GB = get_GB_rescaled(pot, gb)
    caltype = 'Fe_only'
    GB.calc = define_calc(pot, caltype)
    opt = set_optimizer(optimizer, GB)
    opt.run(fmax=fmax, steps=steps)
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_uMLIPs:
        write(f'GBs/Relaxed/{pot}_{gb}_relaxed.vasp', GB)
    if pot in pot_list_EIPs:
        write(f'GBs/Relaxed/{pot}_{gb}_relaxed.lammps-data', GB)


def relax_GB_novac(pot, gb, optimizer, fmax, steps):
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
        write(f'GBs/No_vac_relaxed/{pot}_{gb}_relaxed.vasp', GB)
    if pot in pot_list_EIPs:
        write(f'GBs/No_vac_relaxed/{pot}_{gb}_relaxed.lammps-data', GB)


pot_list = ['NEP_89']
gb_list = ['Sigma37(610)', 'Sigma13(510)', 'Sigma17(410)', 'Sigma5(310)', 'Sigma29(520)', 'Sigma5(210)', 'Sigma13(320)', 'Sigma25(430)', 'Sigma41(540)']
optimizer = 'BFGS'
fmax = 1e-4
steps = 1000
for pot in pot_list:
    for gb in gb_list:
        relax_GB(pot, gb, optimizer, fmax, steps)
