from Functions_BCC import E_bulk_cal, E_bulk_sub_cal, E_gb_cal


pot_list = []
gb_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']
optimizer = 'BFGS'
fmax = 0.0001
steps = 500

for pot in pot_list:
    E_bulk_cal(pot, optimizer, fmax, steps)
    E_bulk_sub_cal(pot, optimizer, fmax, steps)
    for GB in gb_list:
        E_gb_cal(pot, GB, optimizer, fmax, steps)

