import matplotlib.pyplot as plt
import numpy as np
from Mai_plot_setup import get_color, get_data_dict, get_e_list, get_legend

pot_list= ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee', 'DFT']
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']
data_dict = get_data_dict(pot_list, GB_list)


def get_plot_overview(data_dict, plotted_pot_list, GB_list):
    plt.figure(figsize=(8, 8))
    plt.rc('font', family='Times New Roman', size=12)
    legend_handles = set()  # To keep track of unique labels

    for potchoice in plotted_pot_list:
        if potchoice == 'FeCrCoNiCu_Deluigi':
            continue
        else:
            e_list = get_e_list(potchoice)
            color = get_color(potchoice)
            legend = get_legend(potchoice)
            for GBchoice in GB_list:
                for sub_e in e_list:
                    # Plot all points for the current potential and grain boundary
                    # Only add the label for the first occurrence of this potential
                    plt.scatter(data_dict['DFT'][GBchoice][sub_e],
                                data_dict[potchoice][GBchoice][sub_e],
                                marker='o',
                                label=legend if legend not in legend_handles else "",
                                color=color)
                    legend_handles.add(legend)  # Add the current potchoice to the set
    x = np.linspace(-10, 10, 100)
    y = x
    plt.plot(x, y, linestyle=':', color='black')
    plt.xlim(-1.5, 0.1)
    plt.ylim(-1.5, 0.1)
    plt.title(f'BCC-Fe', fontsize=20, pad=20)
    plt.xlabel('DFT value (eV)', fontsize=16, labelpad=5)
    plt.ylabel('uMLIPs value (eV)', fontsize=16, labelpad=5)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc='best',handletextpad=0.05)
    plt.grid(False)
    plt.savefig('Results Analysis/BCC-Fe Lowest Segregation Energy Comparison Overview.pdf', dpi=600)
    #plt.show()

plotted_pot_list= ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
get_plot_overview(data_dict, plotted_pot_list, GB_list)

