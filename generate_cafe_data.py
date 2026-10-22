"""
generate_cafe_data.py
---------------------------------
Generates the synthetic datasets for the course
"Correlation and Relationships in Data" (coffee-shop theme).

Every relationship below is engineered on purpose to demonstrate a specific idea.
Re-run this script to regenerate the CSVs. Change SEED to get a fresh draw
with the same structure.

Outputs:
  - cafe_daily.csv             main daily operations dataset (731 days)
  - cafe_promo_experiment.csv  randomised A/B promo test
  - cafe_two_stores.csv        two-location dataset (Simpson's paradox)
"""

import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------
# 1. MAIN DAILY DATASET  (cafe_daily.csv)
# ----------------------------------------------------------------------
N_DAYS = 731                      # two full years
dates = pd.date_range("2024-01-01", periods=N_DAYS, freq="D")
day_index = np.arange(N_DAYS)     # 0..730, used for the long-term trend
doy = dates.dayofyear.to_numpy()
dow = dates.dayofweek.to_numpy()  # Mon=0 .. Sun=6
is_weekend = (dow >= 5).astype(int)

# --- Temperature: seasonal sine wave (NL-ish, ~0-24 C) ---------------
temperature = 12 + 11 * np.sin(2 * np.pi * (doy - 110) / 365) + rng.normal(0, 1.6, N_DAYS)
temperature = np.round(temperature, 1)

# --- Iced drinks: strong POSITIVE relationship with temperature ------
iced_drink_sales = 20 + 4.6 * temperature + rng.normal(0, 11, N_DAYS)

# --- Hot drinks: strong NEGATIVE relationship with temperature -------
hot_drink_sales = 185 - 3.3 * temperature + rng.normal(0, 12, N_DAYS)

# --- Cold brew & salad: BOTH driven by temperature (confounder) ------
#     They correlate with each other, but neither causes the other.
cold_brew_sales = 6 + 2.3 * temperature + rng.normal(0, 6, N_DAYS)
salad_sales     = 9 + 1.7 * temperature + rng.normal(0, 6, N_DAYS)

# --- Foot traffic: weekend + weather + slow GROWTH over time ---------
foot_traffic = (220 + 55 * is_weekend + 1.6 * temperature
                + 0.12 * day_index + rng.normal(0, 18, N_DAYS))

# --- Barista shoe size: independent noise (NEAR-ZERO / spurious) -----
barista_shoe_size = np.round(rng.normal(42, 1.8, N_DAYS) * 2) / 2  # EU half-sizes

# --- Social followers: monotonic GROWTH over time -------------------
#     Trends up like revenue -> spurious correlation (shared time trend)
social_followers = (500 + 6.5 * day_index
                    + np.cumsum(rng.normal(0, 4, N_DAYS))).round().astype(int)

# --- Promo flag: NON-random, more likely on weekends (confounded) ---
promo_prob = np.where(is_weekend == 1, 0.65, 0.15)
promo_active = (rng.random(N_DAYS) < promo_prob).astype(int)

# --- Discount %: business experiments across a fixed menu of values -
discount_pct = rng.choice([0, 5, 10, 15, 20, 25, 30, 35, 40], size=N_DAYS)

# --- Revenue: built from sales + traffic, carries the upward trend --
daily_revenue = (2.6 * iced_drink_sales + 2.4 * hot_drink_sales
                 + 3.8 * cold_brew_sales + 5.0 * salad_sales
                 + 1.2 * foot_traffic + 40 * promo_active
                 + 0.9 * day_index               # steady business growth over 2 yrs
                 + rng.normal(0, 60, N_DAYS))

# --- Profit: INVERTED-U in discount (Pearson hides this) ------------
rev_centered = daily_revenue - daily_revenue.mean()
daily_profit = (600 + 30 * discount_pct - 0.85 * discount_pct**2
                + 0.05 * rev_centered + rng.normal(0, 40, N_DAYS))

# --- Pastry demand: target for the regression example -----------------
pastry_demand = (30 + 0.16 * foot_traffic + 12 * is_weekend
                 + 0.3 * temperature + rng.normal(0, 8, N_DAYS))

# floor values at zero so nothing is negative
for arr in (iced_drink_sales, hot_drink_sales, cold_brew_sales, salad_sales,
            foot_traffic, daily_revenue, daily_profit, pastry_demand):
    np.clip(arr, 0, None, out=arr)

cafe = pd.DataFrame({
    "date": dates,
    "day_of_week": dates.day_name(),
    "is_weekend": is_weekend,
    "temperature_c": temperature,
    "iced_drink_sales": iced_drink_sales.round().astype(int),
    "hot_drink_sales": hot_drink_sales.round().astype(int),
    "cold_brew_sales": cold_brew_sales.round().astype(int),
    "salad_sales": salad_sales.round().astype(int),
    "foot_traffic": foot_traffic.round().astype(int),
    "barista_shoe_size": barista_shoe_size,
    "social_followers": social_followers,
    "promo_active": promo_active,
    "discount_pct": discount_pct,
    "daily_revenue": daily_revenue.round(2),
    "daily_profit": daily_profit.round(2),
    "pastry_demand": pastry_demand.round().astype(int),
})

# --- Inject one clear OUTLIER day (off-trend catering order) --------
#     Cold day, but a huge one-off iced-drink catering order.
out_idx = 40  # mid-February, low temperature
cafe.loc[out_idx, "iced_drink_sales"] = 380
cafe.loc[out_idx, "daily_revenue"] = round(cafe["daily_revenue"].max() * 1.6, 2)

cafe.to_csv("cafe_daily.csv", index=False)

# ----------------------------------------------------------------------
# 2. RANDOMISED PROMO EXPERIMENT  (cafe_promo_experiment.csv)
# ----------------------------------------------------------------------
#   Promo is assigned RANDOMLY, independent of weekend -> clean causal
#   estimate of the true +40 effect.
N_EXP = 300
exp_dow = rng.integers(0, 7, N_EXP)
exp_weekend = (exp_dow >= 5).astype(int)
exp_promo = rng.integers(0, 2, N_EXP)            # 50/50, independent of weekend
exp_revenue = (1500 + 300 * exp_weekend + 40 * exp_promo
               + rng.normal(0, 35, N_EXP)).round(2)

pd.DataFrame({
    "day": np.arange(1, N_EXP + 1),
    "is_weekend": exp_weekend,
    "promo_active": exp_promo,
    "daily_revenue": exp_revenue,
}).to_csv("cafe_promo_experiment.csv", index=False)

# ----------------------------------------------------------------------
# 3. TWO-STORE DATASET  (cafe_two_stores.csv)  -- Simpson's paradox
# ----------------------------------------------------------------------
#   Within each store: more staff -> higher satisfaction (POSITIVE).
#   Pooled across stores: the relationship REVERSES (NEGATIVE), because
#   the busy downtown store runs high staff AND lower baseline scores.
def store_block(name, n, staff_lo, staff_hi, base):
    staff = rng.integers(staff_lo, staff_hi + 1, n)
    satisfaction = base + 1.2 * staff + rng.normal(0, 1.5, n)
    return pd.DataFrame({
        "store": name,
        "staff_on_shift": staff,
        "customer_satisfaction": satisfaction.round(1),
    })

stores = pd.concat([
    store_block("Downtown", 90, 8, 12, base=55),   # high staff, low base
    store_block("Suburb",   90, 2, 6,  base=80),    # low staff, high base
], ignore_index=True)
stores.to_csv("cafe_two_stores.csv", index=False)

print("Wrote cafe_daily.csv, cafe_promo_experiment.csv, cafe_two_stores.csv")
