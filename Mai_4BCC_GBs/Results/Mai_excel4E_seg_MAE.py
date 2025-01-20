from Mai_plot_setup import get_data_dict, get_legend
import pprint
import pandas as pd

pot_list= ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee', 'DFT']
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']
data_dict = get_data_dict(pot_list, GB_list)

def get_excel4MAE(data):
    with pd.ExcelWriter('Results Analysis/MAE.xlsx', engine='openpyxl') as writer:
        for potchoice, GBchoice in data.items():
            if potchoice == 'FeCrCoNiCu_Deluigi':
                continue
            else:
                name_of_pot = get_legend(potchoice)
                # Create a DataFrame for each potential
                df = pd.DataFrame(GBchoice).T  # Transpose to have GBs as rows
                df.index.name = f'{name_of_pot}'
                # Write the DataFrame to a specific sheet in the Excel file
                df.to_excel(writer, sheet_name=name_of_pot)


get_excel4MAE(data_dict)
