import matplotlib.pyplot as plt
import pandas as pd
import pprint



pot_list = ['MACE', 'CHGNet(v0_3_0)', 'CHGNet(v0_2_0)', 'M3GNet', 'SevenNet', 'DFT']
e_list =  ['Al', 'Si', 'P', 'S', 'Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo', 'W']


data_MACE = [3.6671, 1.5586, 2.1594, 4.8354, -0.5945, -1.5127, -1.3026, -0.5721, 1.1201, 2.5989, 4.5559, -2.0442, -2.317, -4.4587]
data_CHGNetv3 = [3.7254, 1.8329, 1.5618, 3.6565, -0.7144, -1.3307, -1.1668, -0.6865, 1.3186, 2.5938, 4.6251, -2.1182, -2.4141, -4.5945]
data_CHGNetv2 = [3.7279, 2.0165, 1.4076, 4.4016, -0.5769, -1.3277, -1.1134, -0.6475, 1.3554, 2.7692, 4.6607, -2.0186, -2.3261, -4.4887]
data_M3GNet = [4.3849, 2.1751, 2.1888, 4.438, 0.7983, -1.0901, -0.9215, -0.5146, 1.2166, 2.6991, 4.7975, -1.3025, -1.9328, -4.2451]
data_SevenNet = [3.9441, 1.7815, 1.8252, 4.3334, -0.3604, -1.2639, -1.443, -0.7413, 0.95, 2.3407, 4.2218, -1.744, -2.2223, -4.0344]
data_DFT = [3.751, 1.661, 1.842, 4.314, -0.394, -1.453, -1.412, -0.522, 1.069, 2.844, 5.217, -1.992, -2.575, -4.796]

data_list = [data_MACE, data_CHGNetv3, data_CHGNetv2, data_M3GNet, data_SevenNet, data_DFT]
data_dict = {pot: {e_list[i]: data[i] for i in range(len(e_list))} for pot, data in zip(pot_list, data_list)}

df = pd.DataFrame(data_dict).T
df.to_excel('Fe53M.xlsx', index=True)

