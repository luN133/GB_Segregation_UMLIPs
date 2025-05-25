from Functions_FCC import E_bulk_cal, E_bulk_sub_cal, E_gb_cal

pot_list = []
gb_list = ['Sigma37(610)', 'Sigma13(510)', 'Sigma17(410)', 'Sigma5(310)', 'Sigma29(520)', 'Sigma5(210)', 'Sigma13(320)', 'Sigma25(430)', 'Sigma41(540)']
optimizer = 'BFGS'
fmax = 0.0001
steps = 500

for pot in pot_list:
    E_bulk_cal(pot, optimizer, fmax, steps)
    E_bulk_sub_cal(pot, optimizer, fmax, steps)
    for GB in gb_list:
        E_gb_cal(pot, GB, optimizer, fmax, steps)

