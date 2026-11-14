# UPI Data Validation, Volume Forecast and Wallet Fee Study

## 1. Data checks
- Monthly UPI data built for Apr 2016 – Oct 2026 (127 months); RBI data for 22 payment methods, Nov 2019 – Jun 2025 (68 months).
- NPCI and RBI numbers overlap for 65 months and differ by at most 0.41%.
- Yearly totals match the published totals: 2021–22 4,597 crore, 2023–24 13,113 crore (published 13,116), 2025–26 24,162 crore (within 0.03%).

## 2. UPI volume forecast
- Six models were tested from 11 past dates (Mar 2022 – Sep 2024), forecasting 1–24 months ahead.

Average forecast error by months ahead (%):

| Model | 6 | 12 | 18 | 24 |
|---|---|---|---|---|
| SARIMA | 11 | 19 | 30 | 37 |
| Exponential smoothing | 8 | 11 | 18 | 24 |
| LightGBM | 7 | 15 | 25 | 28 |
| Growth model (all years) | 11 | 15 | 27 | 33 |
| S-shaped curve | 17 | 29 | 49 | 62 |
| Growth model (last 3 years) | 12 | 16 | 26 | 31 |

- The error of every model grows the further ahead it forecasts.
- Chosen model: growth model fitted on the last 3 years. It was the most accurate on the recent tests (from late 2023), where the other models predicted too high because UPI growth slowed faster than expected.


Average forecast error 1–2 years ahead, recent tests from September 2023 (%):

| Model | Error |
|---|---|
| Growth model (last 3 years) | **3.1** |
| Growth model (all years) | 22.1 |
| Exponential smoothing | 22.3 |
| LightGBM | 22.6 |
| SARIMA | 25.1 |
| S-shaped curve | 42.2 |

- Chosen model: growth model fitted on the last 3 years. It was by far the most accurate on the recent tests, where the other models predicted too high because UPI growth slowed faster than expected.

Forecast of yearly UPI transactions (crore):

| Year | Low | Expected | High | Growth |
|---|---|---|---|---|
| 2026–27 | 29,220 | 29,417 | 29,624 | 21.8% |
| 2027–28 | 33,524 | 34,225 | 34,973 | 16.3% |
| 2028–29 | 36,696 | 38,150 | 39,803 | 11.5% |
| 2029–30 | 39,131 | 41,471 | 44,410 | 8.7% |
| 2030–31 | 40,872 | **44,171** | 48,659 | 6.5% |

## 3. Wallet fee study 
Wallets were compared with a mix of other payment methods that moved like wallets before April 2023 (1.1% fee on wallet payments above Rs 2,000 from April 2023).

Change in wallets after April 2023, compared with the comparison group:

| Measure | Change |
|---|---|
| Number of payments | +3.2% |
| Value of payments | +6.9% |
| Average payment size | −4.6% |

- Wallet use did not fall after the fee. The changes are no bigger than those of the other payment methods, so there is no sign that the fee reduced wallet use.

## Graphs
- `figures/simple_forecast.png` – monthly UPI transactions with forecast, and yearly growth rate
- `figures/simple_errors.png` – forecast error of each model by months ahead
- `figures/simple_errors_recent.png` – forecast error of each model on the recent tests
- `figures/simple_synth.png` – wallets compared with their comparison group
