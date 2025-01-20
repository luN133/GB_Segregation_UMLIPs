from Ito_plot_setup import get_data_dict, get_legend
import pandas as pd

pot_list= ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee', 'DFT']
GB_list = ['Sigma37(610)', 'Sigma13(510)', 'Sigma17(410)', 'Sigma5(310)', 'Sigma29(520)', 'Sigma5(210)', 'Sigma13(320)', 'Sigma25(430)', 'Sigma41(540)']
data_dict = get_data_dict(pot_list, GB_list)


def get_excel4MAE(data):
    with pd.ExcelWriter('Results Analysis/MAE.xlsx', engine='openpyxl') as writer:
        for potchoice, GBchoice in data.items():
            # Skip specific potchoices
            if potchoice == 'M3GNet':
                continue
            if potchoice == 'FeMnNiCu_Bonny':
                continue
            if potchoice == 'FeTiCoNiCuMoW_Zhou':
                continue
            else:
                name_of_pot = get_legend(potchoice)
                df = pd.DataFrame(GBchoice).T
                df.index.name = f'{name_of_pot}'
                df.to_excel(writer, sheet_name=name_of_pot)


get_excel4MAE(data_dict)
