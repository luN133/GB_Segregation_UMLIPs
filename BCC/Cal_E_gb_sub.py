from mpi4py import MPI
from Functions_BCC import get_e_list, E_gb_sub_cal

# run command
# mpiexec -np 4 python Cal_E_gb_sub.py


# MPI init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set parameters
pot_list = []
gb_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']
e_list = ['Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo', 'W']
optimizer = 'BFGS'
fmax = 0.001
steps = 500

# Distribute tasks
tasks = [(pot, gb, sub_e) for pot in pot_list for gb in gb_list for sub_e in e_list]
my_tasks = tasks[rank::size]

# Loop over each task
for pot, gb, sub_e in my_tasks:
    print(f"\nRank: {rank}   Run ning task: Potential={pot}, GB={gb}, Element={sub_e}\n")
    E_gb_sub_cal(pot, gb, sub_e, optimizer, fmax, steps)
    print(f"Rank: {rank}    Finished task: {pot}, {gb}, {sub_e}\n")
