from ase.io import write, read
from BCC.Functions_BCC import define_calc, set_optimizer, get_pot_list_EIPs,get_pot_list_uMLIPs, get_GB_novac_unrelaxed
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


#pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
pot_list = []
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']
optimizer = 'BFGS'
fmax = 1e-4
for pot in pot_list:
    steps = 1000
    for gb in GB_list:
        relax_GB(pot, gb, optimizer, fmax, steps)
