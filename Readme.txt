###################
File descriptions:

The files of used empirical interatomic potentials are stored in the folder named 'IPs'.

The atomic structures are stored in the folder named 'Structures'.
The GB structures are rescaled according to the lattice parameters for each potential using the python script file named with 'Rescale'.
After being rescaled, they were 'pre-relaxed' by the python script file named with 'Pre-relaxation'. The structures that were 'pre-relaxed' would be used in the main calculation files.


The calculation files are named with 'E_seg' for calculating segregation energy and 'gamma_GB‘ for calculating GB energy, respectively.
Except for the calculation script 'Ito_E_seg_uMLIPs', the other calculation files can be directly used (Because the calculation is not parallel, but the codes ready for parallel calculation is also there.) The calculation script 'Ito_E_seg_uMLIPs' can only be used with parallel calculation. I also prepare a sbatch script for that (but I lost the original file so I don't know if this one can be run successfully)
####################

For the output data, because the output data are stored in separate txt, use the script named with 'E_seg_read4split' in 'Results' folder to pack all the separate data into one txt file '4Data_Analysis'. Then you can use the data to plot the segregation energies. There are some scripts are used by me to generate an excel table and some plots from the data.


Note: The data in '4Data_Analysis.txt' are only lowest segregation energies. The data are in the sequence: [potential [grain boundary [element]]]
potential sequence: ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
GB sequence (Ito as an example): ['Sigma37(610)', 'Sigma13(510)', 'Sigma17(410)', 'Sigma5(310)', 'Sigma29(520)', 'Sigma5(210)', 'Sigma13(320)', 'Sigma25(430)', 'Sigma41(540)']
element sequence: ['Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo']