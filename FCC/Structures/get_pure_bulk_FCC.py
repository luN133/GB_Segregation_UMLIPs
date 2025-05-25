import os
import sys
import numpy as np
from ase.build import make_supercell
from ase.io import write
from ase.build import bulk
from FCC.Functions_FCC import get_lattice_parameter, get_pot_list_EIPs, get_pot_list_uMLIPs



def get_pure_bulk(pot):
    a0_Fe = get_lattice_parameter(pot)
    pure_bulk = bulk('Fe',
                     'fcc',
                     a=a0_Fe,
                     cubic=True)
    multiplier = np.identity(3)*3
    pure_bulk = make_supercell(pure_bulk, multiplier)
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    if pot in pot_list_uMLIPs:
        write(f"Pure_bulks/pure_bulk_{pot}.vasp", pure_bulk)
    if pot in pot_list_EIPs:
        write(f"Pure_bulks/pure_bulk_{pot}.lammps-data", pure_bulk)




pot_list = ['NEP_89']
for pot in pot_list:
    get_pure_bulk(pot)
