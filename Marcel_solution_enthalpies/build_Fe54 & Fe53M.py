import numpy as np
from ase.build import make_supercell, bulk
from ase.io import write



a0_Fe = 2.8592
Fe54 = bulk('Fe', 'bcc', a=a0_Fe, cubic=True)
multiplier = np.identity(3) * 3
Fe54 = make_supercell(Fe54, multiplier)
write('Structures/Fe54.xyz', Fe54)
Fe53M = Fe54.copy()
Fe53M[0].symbol = 'Al'
write('Structures/Fe53M.xyz', Fe53M)