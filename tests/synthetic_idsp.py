"""TEST-ONLY generator of a fake, messy IDSP-style CSV.

This exists only so the loader in validate_risk.py can be tested. It must
never be used to produce any reported result.
"""
import pandas as pd


def make_fake_idsp_csv(path, districts, year=2020):
    rows = []
    week_dates = pd.date_range(f"{year}-06-01", f"{year}-10-31", freq="7D")
    for i, d in enumerate(week_dates):
        district = districts[i % len(districts)]
        # Mess up the spelling on some rows, as in real files.
        shown = district.upper() if i % 3 == 0 else (district[:-1] + "h" if i % 5 == 0 else district)
        rows.append({"Name of State/UT": "Assam" if i % 7 else "ASSAM ",
                     "Name of District": shown,
                     "Disease/ Illness": ["Acute Diarrhoeal Disease", "Cholera", "Measles"][i % 3],
                     "No. of Cases": 10 + i, "No. of Deaths": 0,
                     "Date of start of outbreak": d.strftime("%d/%m/%Y"),
                     "Rainfall (mm)": 100 + 3 * i})
    rows.append({"Name of State/UT": "Bihar", "Name of District": "Patna",
                 "Disease/ Illness": "Cholera", "No. of Cases": 5, "No. of Deaths": 0,
                 "Date of start of outbreak": f"01/07/{year}", "Rainfall (mm)": 50})
    df = pd.DataFrame(rows)
    df = pd.concat([df, df.iloc[:4]], ignore_index=True)  # exact duplicate rows
    df.to_csv(path, index=False)
    return df
