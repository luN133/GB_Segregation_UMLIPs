import matplotlib.pyplot as plt
import numpy as np



#Potentials: ['MACE(large)', 'CHGNet(v0.3.0)', 'CHGNet(v0.2.0)', 'M3GNet(MP-2021.2.8-PES)']
alloying_elements = ['Al', 'Si', 'P', 'S', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo', 'W']


data_MACE_8d = [0.26166875, 0.191575, 0.25033125, 0.3763, -0.0411, -0.08955, -0.08223125, -0.03964375, 0.08610625, 0.1831625, 0.34346875, -0.110425, -0.13705625, -0.25621875]
data_MACE_4c = [0.2599, 0.198, 0.2773, 0.4592, -0.0352, -0.0817, -0.0715, -0.0336, 0.0887, 0.1807, 0.3359, -0.1038, -0.1255, -0.2432]
data_CHGNetv3_8d = [0.26540625, 0.179375, 0.1814, 0.2984, -0.04955, -0.0948, -0.0937625, -0.06210625, 0.08545, 0.17076875, 0.31789375, -0.12969375, -0.1492375, -0.28221875]
data_CHGNetv3_4c = [0.2661, 0.1821, 0.192, 0.351, -0.0468, -0.0913, -0.0888, -0.0606, 0.0828, 0.1669, 0.3154, -0.1222, -0.1466, -0.2766]
data_CHGNetv2_8d = [0.26135, 0.19575, 0.194925, 0.2935625, -0.06569375, -0.0903125, -0.07496875, -0.0576625, 0.0807625, 0.174775, 0.3114, -0.14160625, -0.140275, -0.27436875]
data_CHGNetv2_4c = [0.2664, 0.2075, 0.2186, 0.318, -0.063, -0.0871, -0.073, -0.0532, 0.079, 0.1724, 0.3111, -0.1365, -0.1385, -0.2722]
data_M3GNet_8d = [-0.93828125, 5.36348125, 5.53651875, 0.87903125, 1.99148125, 0.6162375, -0.39780625, 0.4807625, 14.57843125, 0.6982125, 4.03966875, 5.72510625, -1.3726375, -1.399025]
data_M3GNet_4c = [0.8364, -1.0489, 8.1652, 3.0873, -1.2651, -0.0706, -1.3014, 5.398, 0.1998, -1.0341, -0.8985, 1.0667, 2.9683, -0.4968]
data_SevenNet_8d = [0.25916875, 0.1625, 0.19401875, 0.3511375, -0.05349375, -0.09753125, -0.0984, -0.04681875, 0.09028125, 0.16998125, 0.3268, -0.12871875, -0.1523625, -0.2783875]
data_SevenNet_4c = [0.2727, 0.1882, 0.2306, 0.3796, -0.0501, -0.0949, -0.0902, -0.0456, 0.0868, 0.1695, 0.3251, -0.125, -0.1478, -0.2682]


data_DFT_FM_4c = [-0.17,0.54,1.6,2.77,-1.03,-0.63,0.0,0.08,0.09,0.22,0.98,-0.1,0.24,0.28]
data_DFT_FM_8d = [-0.24,0.29,0.39,1.0,-1.04,-0.72,-0.1,0.05,0.04,0.14,0.88,-0.15,0.18,0.2]
data_DFT_NM_4c = [-0.54,0.17,1.35,2.77,-1.41,-0.98,-0.33,-0.19,0.05,0.11,0.79,-0.34,0.01,0.02]
data_DFT_NM_8d = [-0.53,-0.07,0.44,1.04,-1.40,-1.12,-0.51,-0.30,0.13,0.25,0.97,-0.40,-0.11,-0.13]

x = np.arange(len(alloying_elements))


plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'Times New Roman'  # Change 'serif' to your desired font type

plt.plot(x, data_DFT_NM_4c, 'o', color='black', label='4c-NM-DFT', markersize=10)
plt.plot(x, data_DFT_NM_8d, 's', color='black', label='8d-NM-DFT', markersize=10)
plt.plot(x, data_DFT_FM_4c, 'o', color='black', markerfacecolor='none', label='4c-FM-DFT', markersize=10)
plt.plot(x, data_DFT_FM_8d, 's', color='black', markerfacecolor='none', label='8d-FM-DFT', markersize=10)
plt.plot(x, data_MACE_4c, 'o', color='royalblue', label='4c-MACE-MP-0', markersize=10)
plt.plot(x, data_MACE_8d, 's', color='royalblue', label='8d-MACE-MP-0', markersize=10)
plt.plot(x, data_CHGNetv3_4c, 'o', color='green', label='4c-CHGNetv(0.3.0)', markersize=10)
plt.plot(x, data_CHGNetv3_8d, 's', color='green', label='8d-CHGNetv(0.3.0)', markersize=10)
plt.plot(x, data_CHGNetv2_4c, 'o', color='darkorange', label='4c-CHGNetv(0.2.0)', markersize=10)
plt.plot(x, data_CHGNetv2_8d, 's', color='darkorange', label='8d-CHGNetv(0.2.0)', markersize=10)
plt.plot(x, data_M3GNet_8d, 's', color='cyan', label='8d-M3GNet', markersize=10)
plt.plot(x, data_M3GNet_4c, 'o', color='cyan', label='4c-M3GNet', markersize=10)
plt.plot(x, data_SevenNet_8d, 's', color='magenta', label='8d-SevenNet-0', markersize=10)
plt.plot(x, data_SevenNet_4c, 'o', color='magenta', label='4c-SevenNet-0', markersize=10)


plt.ylim(-1.8, 2.1)
plt.xlabel('Alloying Elements', fontsize=16, labelpad=5)
plt.ylabel('$\Delta \mathrm{H_f}$ [eV/atom]', fontsize=16, labelpad=5)
plt.xticks(x, alloying_elements, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc=(1.05, 0.2))
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(True, axis='x', linestyle='--', linewidth=0.5)
plt.savefig('delta_H_cementite.pdf', dpi=600, bbox_inches='tight')

# Displaying the plot
plt.tight_layout()
plt.show()