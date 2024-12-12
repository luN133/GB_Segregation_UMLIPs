import ast


pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
GB_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']


data = []
DFT_data = [[-0.48,-0.11,-0.18,-0.5,-0.18,-0.41,-0.51,-0.77,-0.44,-0.39], [-0.09,0,-0.07,-0.18,-0.04,-0.15,-0.24,-0.15,-0.1,-0.08], [-0.74,-0.12,-0.21,-0.54,-0.14,-0.43,-0.64,-1,-0.43,-0.31], [-0.65,-0.2,-0.29,-0.62,-0.15,-0.56,-0.74,-0.91,-0.55,-0.51]]

for potchoice in pot_list:
    E_seg_pot = []
    for GBchoice in GB_list:
        file_path = f'E_seg_min/{potchoice}_{GBchoice}.txt'
        with open(file_path, 'r') as file:
            contents = file.read()
            line = contents.strip()
            value_list = ast.literal_eval(line.split('=')[1].strip())
        E_seg_pot.append(value_list)
    data.append(E_seg_pot)
data.append(DFT_data)
with open(f"4Data_analysis.txt", "w") as file:
    file.write(f"data_list_tot = {data}\n")
for i, potchoice in enumerate(pot_list):
    with open(f"4Data_analysis.txt", "a") as file:
        file.write(f"{potchoice}_data = {data[i]}\n")