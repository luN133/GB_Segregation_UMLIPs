from ase.io import write
from ase.build import bulk
import numpy as np
from ase.build import make_supercell
from BCC.Functions_BCC import get_lattice_parameter, get_pot_list_uMLIPs, get_pot_list_EIPs

pot_list_uMLIPs = get_pot_list_uMLIPs()
pot_list_EIPs = get_pot_list_EIPs()
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']


def create_pure_bulk(pot):
    pot_list_uMLIPs = get_pot_list_uMLIPs()
    pot_list_EIPs = get_pot_list_EIPs()
    a0_Fe = get_lattice_parameter(pot)
    pure_bulk = bulk('Fe',
                     'bcc',
                     a=a0_Fe,
                     cubic=True)
    multiplier = np.identity(3)*4
    pure_bulk = make_supercell(pure_bulk, multiplier)
    if pot in pot_list_uMLIPs:
        write(f"Pure_bulks/pure_bulk_{pot}.vasp", pure_bulk)
    elif pot in pot_list_EIPs:
        write(f"Pure_bulks/pure_bulk_{pot}.lammps-data", pure_bulk)


pot_list = []
for pot in pot_list:
    create_pure_bulk(pot)





