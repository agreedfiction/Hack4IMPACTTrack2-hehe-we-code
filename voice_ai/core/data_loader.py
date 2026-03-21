import pandas as pd
import os

def load_prices():
    try:
        base = os.path.dirname(os.path.dirname(__file__))
        path = os.path.join(base, "data", "mandi_prices.csv")

        df = pd.read_csv(path)

        prices = {}

        for _, row in df.iterrows():
            commodity = row["commodity"].lower()
            quintal_price = float(row["price"])   # ✅ already quintal

            prices[commodity] = quintal_price     # ✅ store directly

        print("📊 Loaded prices:", prices)

        return prices

    except Exception as e:
        print("⚠️ CSV load failed:", e)
        return None