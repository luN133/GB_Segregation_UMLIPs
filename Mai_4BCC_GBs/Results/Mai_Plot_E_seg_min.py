import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from Mai_plot_setup import get_data_dict, get_e_list, get_color


pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee', 'DFT']
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']
data_dict = get_data_dict(pot_list, GB_list)

#Values that are screened out due to large deviations
data_dict['FeTiCoNiCuMoW_Zhou']['Sigma9(2-21)']['Ti'] = None
data_dict['FeTiCoNiCuMoW_Zhou']['Sigma9(2-21)']['Mo'] = None
data_dict['FeTiCoNiCuMoW_Zhou']['Sigma9(2-21)']['W'] = None


def get_plot(ax, GBchoice, show_legend):

    MACE_data = data_dict['MACE'][GBchoice]
    CHGNet_v0_3_0_data = data_dict['CHGNet(v0_3_0)'][GBchoice]
    CHGNet_v0_2_0_data = data_dict['CHGNet(v0_2_0)'][GBchoice]
    M3GNet_data = data_dict['M3GNet'][GBchoice]
    SevenNet_data = data_dict['SevenNet'][GBchoice]
    FeMnNiCu_Bonny_data = data_dict['FeMnNiCu_Bonny'][GBchoice]
    FeCrCoNiCu_Deluigi_data = data_dict['FeCrCoNiCu_Deluigi'][GBchoice]
    FeTiCoNiCuMoW_Zhou_data = data_dict['FeTiCoNiCuMoW_Zhou'][GBchoice]
    FeCu_Lee_data = data_dict['FeCu_Lee'][GBchoice]
    DFT_data = data_dict['DFT'][GBchoice]

    e_list_uMLIPs = ['Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo', 'W']
    e_list_Bonny = get_e_list('FeMnNiCu_Bonny')
    e_list_Deluigi = get_e_list('FeCrCoNiCu_Deluigi')
    e_list_Zhou = get_e_list('FeTiCoNiCuMoW_Zhou')
    e_list_Lee = get_e_list('FeCu_Lee')

    MACE_values = list(MACE_data.values())
    CHGNet_v0_3_0_values = list(CHGNet_v0_3_0_data.values())
    CHGNet_v0_2_0_values = list(CHGNet_v0_2_0_data.values())
    M3GNet_values = list(M3GNet_data.values())
    SevenNet_values = list(SevenNet_data.values())
    FeMnNiCu_Bonny_values = list(FeMnNiCu_Bonny_data.values())
    FeCrCoNiCu_Deluigi_values = list(FeCrCoNiCu_Deluigi_data.values())
    FeTiCoNiCuMoW_Zhou_values = list(FeTiCoNiCuMoW_Zhou_data.values())
    FeCu_Lee_values = list(FeCu_Lee_data.values())
    dft_values = list(DFT_data.values())

    # Plot each dataset with markers
    ax.plot(e_list_uMLIPs, MACE_values, marker='o', label='MACE-MP-0', linestyle='-', color=get_color('MACE'))
    ax.plot(e_list_uMLIPs, CHGNet_v0_3_0_values, marker='o', label='CHGNet(v0.3.0)', linestyle='-',
            color=get_color('CHGNet(v0_3_0)'))
    ax.plot(e_list_uMLIPs, CHGNet_v0_2_0_values, marker='o', label='CHGNet(v0.2.0)', linestyle='-',
            color=get_color('CHGNet(v0_2_0)'))
    ax.plot(e_list_uMLIPs, M3GNet_values, marker='o', label='M3GNet', linestyle='-', color=get_color('M3GNet'))
    ax.plot(e_list_uMLIPs, SevenNet_values, marker='o', label='SevenNet-0', linestyle='-', color=get_color('SevenNet'))
    ax.plot(e_list_Bonny, FeMnNiCu_Bonny_values, marker='o', label='FeMnNiCu_Bonny', linestyle='-',
            color=get_color('FeMnNiCu_Bonny'))
    ax.plot(e_list_Zhou, FeTiCoNiCuMoW_Zhou_values, marker='o', label='FeTiCoNiCuMoW_Zhou', linestyle='-',
            color=get_color('FeTiCoNiCuMoW_Zhou'))
    ax.scatter(e_list_Lee, FeCu_Lee_values, marker='o', label='FeCu_Lee', linestyle='-', color=get_color('FeCu_Lee'))
    ax.plot(e_list_uMLIPs, dft_values, marker='X', label='DFT', linestyle=':', color=get_color('DFT'))

    # Add title and labels
    GBchoice_with_sigma = GBchoice.replace("Sigma", "$\\Sigma$")
    ax.set_title(f'BCC-Fe {GBchoice_with_sigma}', fontsize=16, pad=8)
    #ax.set_xlabel('Solute Element', fontsize=14)
    #ax.set_ylabel('Segregation Energy (eV)', fontsize=14)
    ax.xaxis.grid(True, linestyle='--', linewidth=0.7)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.1f}'))

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        return handles, labels
    else:
        ax.legend().set_visible(False)
        return None, None


# Create the figure and axes for subplots
plt.rcParams['font.family'] = 'Times New Roman'
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))
plt.subplots_adjust(hspace=0.2, wspace=0.1, bottom=0.2)  # Adjust bottom to make room for the legend
fig.text(0.075, 0.5, 'Segregation Energy (eV)', va='center', rotation='vertical', fontsize=16)
# Collect handles and labels from one subplot to create a unified legend
all_handles = []
all_labels = []

for i, GBchoice in enumerate(GB_list):
    row = i // 2
    col = i % 2
    show_legend = (i == 0)  # Only show legend in the first subplot
    handles, labels = get_plot(axs[row, col], GBchoice, show_legend)
    if handles:
        all_handles.extend(handles)
        all_labels.extend(labels)

# Place the legend at the bottom of the figure, spanning the width of the figure
fig.legend(all_handles, all_labels, loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=5,
           fontsize=14, frameon=True)

plt.savefig('Results Analysis/bcc-Fe E_seg_4GBs.pdf', dpi=1000, bbox_inches='tight')
plt.show()

