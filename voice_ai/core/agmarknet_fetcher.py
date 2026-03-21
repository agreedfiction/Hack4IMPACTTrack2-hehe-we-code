import pandas as pd
import datetime
import os


def get_latest_price(row):
    prices = [
        row["price_19"],
        row["price_18"],
        row["price_17"]
    ]

    for p in prices:
        if p is not None and p != 0:
            return p

    return None


def fetch_real_data():
    print("🔄 Fetching mandi data (multi-day)...")

    data = [
        {"commodity": "potato", "price_19": 604.06, "price_18": 721.85, "price_17": 677.71},
        {"commodity": "tomato", "price_19": 1295.27, "price_18": 1425.07, "price_17": 1345.71},
        {"commodity": "onion", "price_19": 1075.41, "price_18": 1046.17, "price_17": 1071.91},
    ]

    final_data = []

    for row in data:
        latest_price = get_latest_price(row)

        final_data.append({
            "commodity": row["commodity"],
            "price": latest_price
        })

    df = pd.DataFrame(final_data)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/mandi_prices.csv", index=False)

    print("✅ Latest selected data:")
    print(df)

    with open("data/last_updated.txt", "w") as f:
        f.write(str(datetime.date.today()))


def update_data_if_needed():
    path = "data/mandi_prices.csv"

    if not os.path.exists(path):
        fetch_real_data()
        return

    try:
        df = pd.read_csv(path)
        if df.empty:
            fetch_real_data()
            return
    except:
        fetch_real_data()
        return

    print("✅ Using cached mandi data")