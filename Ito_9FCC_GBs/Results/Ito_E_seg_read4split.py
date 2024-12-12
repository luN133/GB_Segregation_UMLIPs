import ast

pot_list_uMLIPs = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet']
pot_list_EIPs = ['FeMnNiCu_Bonny', 'FeCrCoNiCu_Deluigi', 'FeTiCoNiCuMoW_Zhou', 'FeCu_Lee']
GB_list = ['Sigma37(610)', 'Sigma13(510)', 'Sigma17(410)', 'Sigma5(310)',
           'Sigma29(520)', 'Sigma5(210)', 'Sigma13(320)', 'Sigma25(430)', 'Sigma41(540)']
e_list = ['Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo']
DFT_data = [[-0.8943, -0.2193, 0.0607, -0.1469, -0.0751, -0.068, -0.215, -1.2457, -0.6185], [-0.9381, -0.1831, 0.022, -0.1756, -0.1116, -0.0837, -0.2296, -1.2502, -0.6203], [-0.8947, -0.2463, 0.0288, -0.1669, -0.1013, -0.0497, -0.2264, -1.2012, -0.6048], [-0.7923, -0.2328, 0.0274, -0.0979, -0.1073, -0.0493, -0.1229, -1.0943, -0.58], [-0.7727, -0.3302, -0.0322, -0.1101, -0.0868, -0.0452, -0.1448, -1.0226, -0.6547], [-0.6669, -0.3487, 0.003, 0.0479, -0.0485, 0.0029, -0.1112, -0.8446, -0.5338], [-0.5059, -0.1006, 0.0931, 0.0036, -0.0511, -0.0246, -0.2412, -0.8012, -0.466], [-0.5923, -0.1661, 0.0432, -0.0469, -0.0731, -0.025, -0.1872, -0.8805, -0.5059], [-0.6066, -0.1806, 0.0206, -0.0525, -0.0925, -0.0725, -0.2255, -0.8928, -0.5211]]

data = []

for potchoice in pot_list_uMLIPs:
    E_seg_pot = []
    for GBchoice in GB_list:
        E_seg_GB = []
        for element in e_list:
            file_path = f'E_seg_min/uMLIPs/{potchoice}_{GBchoice}_{element}.txt'
            with open(file_path, 'r') as file:
                contents = file.read()
                line = contents.strip()
                value = float(line.split('=')[1].strip())
            E_seg_GB.append(value)
        E_seg_pot.append(E_seg_GB)
    data.append((E_seg_pot))


for potchoice in pot_list_EIPs:
    E_seg_pot = []
    for GBchoice in GB_list:
        file_path = f'E_seg_min/EIPs_FIRE/{potchoice}_{GBchoice}.txt'
        with open(file_path, 'r') as file:
            contents = file.read()
            line = contents.strip()
            value_list = ast.literal_eval(line.split('=')[1].strip())
        E_seg_pot.append(value_list)
    data.append(E_seg_pot)
data.append(DFT_data)

with open(f"4Data_analysis.txt", "w") as file:
    file.write(f"data_list_tot = {data}\n")
for i, potchoice in enumerate(pot_list_uMLIPs):
    with open(f"4Data_analysis.txt", "a") as file:
        file.write(f"{potchoice}_data = {data[i]}\n")
for i, potchoice in enumerate(pot_list_EIPs):
    with open(f"4Data_analysis.txt", "a") as file:
        file.write(f"{potchoice}_data = {data[i]}\n")