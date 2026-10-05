"""
Sales Data Analysis - Pandas, NumPy & Matplotlib Mini Project

Steps:
  1. Create a sample (deliberately messy) sales dataset with NumPy
  2. Load and explore it with Pandas
  3. Clean it (duplicates, missing values, inconsistent text)
  4. Analyze it (totals, groupings, trends)
  5. Visualize it (bar, line, pie charts) with Matplotlib

Run with:  python sales_analysis.py
Requires:  pip install pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # lets charts be saved without a display; remove if you want plt.show()
import matplotlib.pyplot as plt

CSV_FILE = "sales_data.csv"


# =====================================================
# 1. CREATE A SAMPLE DATASET
# =====================================================
def create_sample_data(n=300, seed=42):
    rng = np.random.default_rng(seed)

    products = {          # product: (category, unit price)
        "Laptop": ("Electronics", 900),
        "Phone": ("Electronics", 600),
        "Headphones": ("Electronics", 80),
        "Desk": ("Furniture", 250),
        "Chair": ("Furniture", 150),
        "Notebook": ("Stationery", 5),
        "Pen Set": ("Stationery", 12),
    }
    names = rng.choice(list(products), size=n)

    df = pd.DataFrame({
        "order_id": np.arange(1001, 1001 + n),
        "date": pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
        "product": names,
        "category": [products[p][0] for p in names],
        "unit_price": [products[p][1] for p in names],
        "quantity": rng.integers(1, 8, n).astype(float),
        "region": rng.choice(["North", "South", "East", "West"], size=n),
    })

    # Make the data messy on purpose so we have something to clean
    df.loc[rng.choice(n, 15, replace=False), "quantity"] = np.nan          # missing values
    df.loc[rng.choice(n, 10, replace=False), "region"] = np.nan            # missing values
    df.loc[rng.choice(n, 20, replace=False), "region"] = "  north "        # inconsistent text
    df = pd.concat([df, df.sample(8, random_state=1)], ignore_index=True)  # duplicate rows

    df.to_csv(CSV_FILE, index=False)
    print(f"Sample dataset saved to {CSV_FILE}")


# =====================================================
# 2. LOAD & EXPLORE
# =====================================================
def explore(df):
    print("\n--- First 5 rows ---")
    print(df.head())
    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\n--- Data types ---")
    print(df.dtypes)
    print("\n--- Missing values per column ---")
    print(df.isna().sum())
    print(f"\nDuplicate rows: {df.duplicated().sum()}")
    print("\n--- Summary statistics ---")
    print(df.describe())


# =====================================================
# 3. CLEAN
# =====================================================
def clean(df):
    before = len(df)

    df = df.drop_duplicates()                                   # remove duplicate rows

    df["region"] = df["region"].str.strip().str.title()         # fix "  north " -> "North"
    df["region"] = df["region"].fillna("Unknown")               # fill missing text

    median_qty = df["quantity"].median()                        # fill missing numbers
    df["quantity"] = df["quantity"].fillna(median_qty).astype(int)

    df["date"] = pd.to_datetime(df["date"])                     # make sure dates are real dates
    df["revenue"] = df["unit_price"] * df["quantity"]           # new calculated column
    df["month"] = df["date"].dt.to_period("M").astype(str)

    print(f"\nCleaning done: {before} -> {len(df)} rows "
          f"(quantity filled with median = {median_qty:.0f})")
    return df


# =====================================================
# 4. ANALYZE
# =====================================================
def analyze(df):
    results = {}

    print("\n=== Key numbers ===")
    print(f"Total revenue:       {df['revenue'].sum():,.0f}")
    print(f"Average order value: {df['revenue'].mean():,.2f}")
    print(f"Median order value:  {np.median(df['revenue']):,.2f}")
    print(f"Biggest order:       {df['revenue'].max():,.0f}")

    results["by_product"] = df.groupby("product")["revenue"].sum().sort_values(ascending=False)
    results["by_category"] = df.groupby("category")["revenue"].sum()
    results["by_month"] = df.groupby("month")["revenue"].sum()
    results["by_region"] = df.groupby("region")["revenue"].sum()

    print("\n=== Revenue by product ===")
    print(results["by_product"].round(0))
    print("\n=== Revenue by region ===")
    print(results["by_region"].round(0))

    best = results["by_month"].idxmax()
    print(f"\nBest month: {best} ({results['by_month'][best]:,.0f})")
    return results


# =====================================================
# 5. VISUALIZE
# =====================================================
def make_charts(results):
    # Bar chart: revenue by product
    ax = results["by_product"].plot(kind="bar", color="steelblue", figsize=(8, 5))
    ax.set_title("Total Revenue by Product")
    ax.set_xlabel("Product")
    ax.set_ylabel("Revenue")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("bar_revenue_by_product.png", dpi=120)
    plt.close()

    # Line chart: monthly revenue trend
    monthly = results["by_month"]
    plt.figure(figsize=(8, 5))
    plt.plot(monthly.index, monthly.values, marker="o", color="darkorange")
    plt.title("Monthly Revenue Trend")
    plt.xlabel("Month")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45, ha="right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("line_monthly_revenue.png", dpi=120)
    plt.close()

    # Pie chart: revenue share by category
    plt.figure(figsize=(6, 6))
    plt.pie(results["by_category"], labels=results["by_category"].index,
            autopct="%1.1f%%", startangle=90)
    plt.title("Revenue Share by Category")
    plt.tight_layout()
    plt.savefig("pie_category_share.png", dpi=120)
    plt.close()

    print("\nCharts saved: bar_revenue_by_product.png, line_monthly_revenue.png, pie_category_share.png")


# =====================================================
# MAIN
# =====================================================
def main():
    create_sample_data()
    df = pd.read_csv(CSV_FILE)       # load the CSV like you would any real dataset
    explore(df)
    df = clean(df)
    results = analyze(df)
    make_charts(results)
    df.to_csv("sales_data_clean.csv", index=False)


if __name__ == "__main__":
    main()
