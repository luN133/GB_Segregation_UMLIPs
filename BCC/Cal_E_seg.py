import os
import csv
import pandas as pd


pot_list = []
gb_list = ['Sigma3(1-11)', 'Sigma3(1-12)', 'Sigma9(2-21)', 'Sigma11(3-32)']
e_list = ['Ti', 'V', 'Cr', 'Mn', 'Co', 'Ni', 'Cu', 'Nb', 'Mo', 'W']

def E_seg_cal(pot_list, gb_list, e_list):
    for pot in pot_list:
        base_path = os.path.join("Results", pot)

        E_bulk_path = os.path.join(base_path, "E_bulk", f"E_bulk_{pot}.csv")
        E_bulk = pd.read_csv(E_bulk_path)["E_bulk (eV)"].iloc[0]

        for gb in gb_list:
            E_gb_path = os.path.join(base_path, "E_gb", f"E_gb_{pot}_{gb}.csv")
            E_gb = pd.read_csv(E_gb_path)["E_gb (eV)"].iloc[0]

            for sub_e in e_list:
                E_bulk_sub_path = os.path.join(base_path, "E_bulk_sub", f"E_bulk_sub_{pot}_{sub_e}.csv")
                E_bulk_sub = pd.read_csv(E_bulk_sub_path)["E_bulk_sub (eV)"].iloc[0]

                E_gb_sub_path = os.path.join(base_path, "E_gb_sub", f"E_gb_sub_{pot}_{gb}_{sub_e}.csv")
                df = pd.read_csv(E_gb_sub_path)

                out_dir = os.path.join(base_path, "E_seg")
                os.makedirs(out_dir, exist_ok=True)
                output_path = os.path.join(out_dir, f"E_seg_{pot}_{gb}_{sub_e}.csv")

                with open(output_path, "w", newline='') as f_out:
                    writer = csv.writer(f_out)
                    writer.writerow([
                        "Potential", "GB", "Element", "E_seg (eV)", "Site",
                        "Distance_to_GB (Angstrom)", "Fmax", "n_steps", "MaxSteps", "Optimizer"
                    ])

                    for _, row in df.iterrows():
                        E_gb_sub = row["E_gb_sub (eV)"]
                        E_seg = (E_gb_sub - E_gb) - (E_bulk_sub - E_bulk)
                        writer.writerow([
                            pot, gb, sub_e, E_seg,
                            row["Site"],
                            row["Distance to GB (Angstrom)"],
                            row["Fmax"],
                            row["n_steps"],
                            row["MaxSteps"],
                            row["Optimizer"]
                        ])

                print(f" E_seg saved: {output_path}")


def find_E_seg_min(pot_list, gb_list, e_list):
    base_path = "Results"
    all_summary_rows = []

    for pot in pot_list:
        E_seg_min_path = os.path.join(base_path, f"{pot}/E_seg_min_{pot}.csv")
        rows = []
        summary_data = []

        for gb in gb_list:
            row = {"Potential": pot, "GB": gb}
            for sub_e in e_list:
                E_seg_path = os.path.join(base_path, pot, "E_seg", f"E_seg_{pot}_{gb}_{sub_e}.csv")
                df = pd.read_csv(E_seg_path)

                converged = df[df["n_steps"] < df["MaxSteps"]]
                if converged.empty:
                    row[sub_e] = None  # 或 float("nan")
                    continue

                min_row = converged.loc[converged["E_seg (eV)"].idxmin()]
                rows.append(min_row)
                row[sub_e] = min_row["E_seg (eV)"]

            summary_data.append(row)

        # Save per-pot E_seg_min
        if rows:
            df_min = pd.DataFrame(rows)
            df_min.to_csv(E_seg_min_path, index=False)
            print(f"[✓] E_seg_min saved: {E_seg_min_path}")

        # Accumulate summary blocks
        pot_summary_df = pd.DataFrame(summary_data)
        all_summary_rows.append(pot_summary_df)

    # Combine and export final summary
    combined_df = pd.concat(all_summary_rows, ignore_index=True)
    summary_path = os.path.join(base_path, "Summary.csv")
    combined_df.to_csv(summary_path, index=False)
    print(f"[✓] Combined Summary table saved: {summary_path}")





E_seg_cal(pot_list, gb_list, e_list)
find_E_seg_min(pot_list, gb_list, e_list)
