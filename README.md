# UPI Data Validation, Volume Forecast and Wallet Fee Study

## What is in this folder

| File | What it is |
|---|---|
| `src\common.py` | Shared helper (folders, graph style, saving) |
| `src\s00_build_data.py` | Build and check the data |
| `src\s01_forecast.py` | Forecast UPI transactions to 2030-31 |
| `src\s02_fee_response.py` | The April 2023 wallet fee study |
| `data\raw\npci_upi_stats.csv` | Monthly UPI data from NPCI (Apr 2016 onwards) |
| `data\raw\rbi_old.csv` | RBI monthly data for 22 payment methods |
| `data\manual\upi_monthly_recent.csv` | UPI months Jul 2025 - Oct 2026, with sources |
| `requirements.txt` | Python packages needed |
| `run_analysis.bat` | Runs all three steps and saves the results |
| `SUMMARY.md` | Summary of the work and results |

After running it, results are saved in:

- `logs\`
- `figures\`
- `results\`

## One-time setup

Needs Python 3.10 or newer from [python.org](https://www.python.org).

1. Unzip the folder.
2. Open PowerShell and run:

```
cd C:\BTP_upi-main
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

In the same PowerShell window:

```
.\run_analysis.bat
```
