import matplotlib.pyplot as plt
import numpy as np
from Ito_plot_setup import get_color, get_data_dict, get_e_list, get_legend

pot_list= ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee', 'DFT']
GB_list = ['Sigma37(610)', 'Sigma13(510)', 'Sigma17(410)', 'Sigma5(310)', 'Sigma29(520)', 'Sigma5(210)', 'Sigma13(320)', 'Sigma25(430)', 'Sigma41(540)']
data_dict = get_data_dict(pot_list, GB_list)
#print(data_dict['CHGNet(v0_2_0)']['Sigma37(610)'])

def get_plot_overview(data_dict, plotted_pot_list):
    GB_list = ['Sigma37(610)', 'Sigma13(510)', 'Sigma17(410)', 'Sigma5(310)', 'Sigma29(520)',
               'Sigma5(210)', 'Sigma13(320)', 'Sigma25(430)', 'Sigma41(540)']

    # Create the figure before plotting
    plt.figure(figsize=(8, 8))
    plt.rc('font', family='Times New Roman', size=12)
    legend_handles = set()  # To keep track of unique labels

    for potchoice in plotted_pot_list:
        legend = get_legend(potchoice)
        color = get_color(potchoice)
        e_list = get_e_list(potchoice)
        for GBchoice in GB_list:
            if potchoice == 'CHGNet(v0_2_0)': #filter the unconverged result of CHGNet(v0.2.0)
                if GBchoice == 'Sigma37(610)':
                    e_list = ['Ti', 'V', 'Cr', 'Mn', 'Ni', 'Cu', 'Nb', 'Mo']
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
    plt.xlim(-1.5, 0.2)
    plt.ylim(-1.5, 0.2)
    plt.title(f'FCC-Fe', fontsize=20, pad=20)
    plt.xlabel('DFT value (eV)', fontsize=16, labelpad=5)
    plt.ylabel('uMLIPs value (eV)', fontsize=16, labelpad=5)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=10, loc='best',handletextpad=0.05)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('Results Analysis/FCC-Fe Lowest Segregation Energy Comparison Overview.pdf', dpi=600)
    #plt.show()


plotted_pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)']
get_plot_overview(data_dict, plotted_pot_list)

