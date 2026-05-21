# ================================================================
#  Future Interns — ML Task 1 (2026)
#  data.py — Generate Superstore-style Dataset & Save as CSV
# ================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

print("=" * 60)
print("  Future Interns — ML Task 1 (2026)")
print("  data.py — Superstore Dataset Generator")
print("=" * 60)

# ================================================================
# SECTION 1 — REFERENCE TABLES
# ================================================================

STATE_REGION = {
    "Alabama": "South", "Arizona": "West", "Arkansas": "South",
    "California": "West", "Colorado": "West", "Connecticut": "East",
    "Delaware": "East", "District of Columbia": "East", "Florida": "South",
    "Georgia": "South", "Idaho": "West", "Illinois": "Central",
    "Indiana": "Central", "Iowa": "Central", "Kansas": "Central",
    "Kentucky": "South", "Louisiana": "South", "Maine": "East",
    "Maryland": "East", "Massachusetts": "East", "Michigan": "Central",
    "Minnesota": "Central", "Mississippi": "South", "Missouri": "Central",
    "Montana": "West", "Nebraska": "Central", "Nevada": "West",
    "New Hampshire": "East", "New Jersey": "East", "New Mexico": "West",
    "New York": "East", "North Carolina": "South", "North Dakota": "Central",
    "Ohio": "East", "Oklahoma": "Central", "Oregon": "West",
    "Pennsylvania": "East", "Rhode Island": "East", "South Carolina": "South",
    "South Dakota": "Central", "Tennessee": "South", "Texas": "Central",
    "Utah": "West", "Vermont": "East", "Virginia": "South",
    "Washington": "West", "West Virginia": "East", "Wisconsin": "Central",
    "Wyoming": "West",
}

STATE_CITIES = {
    "Alabama": ["Decatur","Montgomery","Florence","Mobile","Auburn"],
    "Arizona": ["Gilbert","Phoenix","Scottsdale","Tucson","Mesa"],
    "Arkansas": ["Fayetteville","Jonesboro","Little Rock","Hot Springs","Texarkana"],
    "California": ["Los Angeles","San Francisco","Roseville","Pasadena","San Jose","San Diego","Sacramento","Fresno","Oakland","Bakersfield"],
    "Colorado": ["Aurora","Denver","Colorado Springs","Arvada","Louisville"],
    "Connecticut": ["Fairfield","Manchester","Norwich","Middletown","Meriden"],
    "Delaware": ["Dover","Wilmington","Newark"],
    "District of Columbia": ["Washington"],
    "Florida": ["Fort Lauderdale","Melbourne","Tampa","Tamarac","Saint Petersburg","Miami","Orlando","Jacksonville"],
    "Georgia": ["Columbus","Atlanta","Warner Robins","Roswell","Macon"],
    "Idaho": ["Meridian","Boise","Lewiston","Pocatello","Caldwell"],
    "Illinois": ["Naperville","Chicago","Orland Park","Bloomington","Decatur","Springfield","Rockford"],
    "Indiana": ["New Albany","Columbus","Richmond","La Porte","Indianapolis"],
    "Iowa": ["Urbandale","Des Moines","Burlington","Cedar Rapids","Dubuque"],
    "Kansas": ["Olathe","Overland Park","Manhattan","Wichita","Garden City"],
    "Kentucky": ["Henderson","Richmond","Louisville","Florence","Murray"],
    "Louisiana": ["Monroe","Bossier City","Lafayette","Lake Charles","Kenner","New Orleans"],
    "Maine": ["Bangor","Lewiston"],
    "Maryland": ["Columbia","Clinton","Rockville","Baltimore","Gaithersburg"],
    "Massachusetts": ["Lowell","Franklin","Lawrence","New Bedford","Everett","Boston"],
    "Michigan": ["Westland","Jackson","Saginaw","Detroit","Taylor","Grand Rapids"],
    "Minnesota": ["Eagan","Rochester","Minneapolis","Saint Paul","Lakeville"],
    "Mississippi": ["Jackson","Gulfport","Southaven","Hattiesburg"],
    "Missouri": ["Independence","Gladstone","Jefferson City","Saint Peters","Springfield"],
    "Montana": ["Great Falls","Missoula","Bozeman","Billings","Helena"],
    "Nebraska": ["Fremont","Omaha","Grand Island","Norfolk"],
    "Nevada": ["Las Vegas","Reno","North Las Vegas","Henderson","Sparks"],
    "New Hampshire": ["Concord","Dover","Nashua"],
    "New Jersey": ["Westfield","Morristown","Belleville","Lakewood","Hackensack"],
    "New Mexico": ["Carlsbad","Farmington","Albuquerque","Las Cruces","Santa Fe"],
    "New York": ["New York City","Troy","New Rochelle","Auburn","Lindenhurst","Buffalo","Albany"],
    "North Carolina": ["Concord","Durham","Charlotte","Chapel Hill","Wilmington"],
    "North Dakota": ["Fargo"],
    "Ohio": ["Columbus","Newark","Hamilton","Akron","Medina","Cleveland","Cincinnati"],
    "Oklahoma": ["Edmond","Norman","Tulsa","Muskogee","Oklahoma City"],
    "Oregon": ["Portland","Salem","Tigard","Redmond","Medford"],
    "Pennsylvania": ["Philadelphia","Chester","Lancaster","Allentown","Reading","Pittsburgh"],
    "Rhode Island": ["Warwick","Providence","Woonsocket","Cranston"],
    "South Carolina": ["Columbia","Florence","North Charleston","Summerville","Mount Pleasant"],
    "South Dakota": ["Sioux Falls","Rapid City","Aberdeen"],
    "Tennessee": ["Memphis","Bristol","Franklin","Columbia","Murfreesboro","Nashville"],
    "Texas": ["Fort Worth","Houston","Richardson","San Antonio","Grand Prairie","Dallas","Austin"],
    "Utah": ["West Jordan","Orem","Layton","Provo","Pleasant Grove","Salt Lake City"],
    "Vermont": ["Burlington"],
    "Virginia": ["Springfield","Arlington","Waynesboro","Richmond","Alexandria"],
    "Washington": ["Seattle","Des Moines","Marysville","Vancouver","Edmonds","Spokane","Tacoma"],
    "West Virginia": ["Wheeling"],
    "Wisconsin": ["Madison","Franklin","Green Bay","Milwaukee","Appleton"],
    "Wyoming": ["Cheyenne"],
}

CAT_SUBCAT = {
    "Furniture":       ["Bookcases","Chairs","Tables","Furnishings"],
    "Office Supplies": ["Labels","Storage","Art","Binders","Appliances","Paper","Envelopes","Fasteners","Supplies"],
    "Technology":      ["Phones","Accessories","Machines","Copiers"],
}

PRODUCTS = {
    "Bookcases":   [("FUR-BO-10001798","Bush Somerset Collection Bookcase"),
                    ("FUR-BO-10004834","Riverside Palais Royal Lawyers Bookcase"),
                    ("FUR-BO-10003396","Sauder Classic Bookcase, Traditional Cherry"),
                    ("FUR-BO-10002213","O'Sullivan 4-Shelf Bookcase in Odessa Pine")],
    "Chairs":      [("FUR-CH-10000454","Hon Deluxe Fabric Upholstered Stacking Chairs"),
                    ("FUR-CH-10002774","Global Deluxe Stacking Chair, Gray"),
                    ("FUR-CH-10001313","SAFCO PVC Mobile Pedestal Chair"),
                    ("FUR-CH-10003473","Harbour Creations 67200 Series Folding Chair")],
    "Tables":      [("FUR-TA-10000577","Bretford CR4500 Series Slim Rectangular Table"),
                    ("FUR-TA-10001539","Chromcraft Rectangular Conference Tables"),
                    ("FUR-TA-10004828","Bevis Round Table, 36\" Diameter"),
                    ("FUR-TA-10002199","DMI Eclipse Executive Suite Corner Table")],
    "Furnishings": [("FUR-FU-10001487","Eldon Expressions Wood Desk Accessories"),
                    ("FUR-FU-10004848","Howard Miller 13-3/4\" Diameter Wall Clock"),
                    ("FUR-FU-10003447","Tensor Brushed Steel Torchiere Floor Lamp"),
                    ("FUR-FU-10002390","Deflect-o DuraMat Chair Mat for Low Pile")],
    "Labels":      [("OFF-LA-10000240","Self-Adhesive Address Labels by Universal"),
                    ("OFF-LA-10002762","Avery 485"),("OFF-LA-10003559","Avery 510"),
                    ("OFF-LA-10004099","Avery Color Coding Labels")],
    "Storage":     [("OFF-ST-10000760","Eldon Fold 'N Roll Cart System"),
                    ("OFF-ST-10004186","Stur-D-Stor Shelving, Vertical 5-Shelf"),
                    ("OFF-ST-10002927","Akro-Mils 09516 Super-Size AkroBins"),
                    ("OFF-ST-10001806","Advantus Super Stacker Divided Storage Box")],
    "Art":         [("OFF-AR-10002833","Newell 322"),("OFF-AR-10003056","Newell 341"),
                    ("OFF-AR-10001986","Sanford 52602 Liquid Accent Tank-Style Highlighters"),
                    ("OFF-AR-10000397","BIC Brite Liner Highlighters, Assorted Colors")],
    "Binders":     [("OFF-BI-10003910","DXL Angle-View Binders with Locking Rings"),
                    ("OFF-BI-10003656","Fellowes PB200 Plastic Comb Binding Machine"),
                    ("OFF-BI-10001602","Wilson Jones Easy Grip Round Ring View Binder"),
                    ("OFF-BI-10004738","GBC DocuBind TL300 Electric Binding System")],
    "Appliances":  [("OFF-AP-10002892","Belkin F5C206VTEL 6 Outlet Surge"),
                    ("OFF-AP-10002311","Holmes Replacement HEPA Filter"),
                    ("OFF-AP-10001028","Cuisinart Classic 12-Cup Coffeemaker"),
                    ("OFF-AP-10000304","Hoover WindTunnel 3 High Performance Pet")],
    "Paper":       [("OFF-PA-10002365","Xerox 1967"),("OFF-PA-10000249","Easy-staple paper"),
                    ("OFF-PA-10003695","Avery Shipping Labels with TrueBlock Technology"),
                    ("OFF-PA-10001784","Hammermill Recycled Paper, 500 Sheet Ream")],
    "Envelopes":   [("OFF-EN-10001509","Poly String Tie Envelopes"),
                    ("OFF-EN-10002986","#10-4 1/8\" Premium Diagonal Seam Envelopes"),
                    ("OFF-EN-10004032","Wausau Papers Astrobrights Envelopes"),
                    ("OFF-EN-10000822","Jiffy Padded Mailers, Multi-Pack")],
    "Fasteners":   [("OFF-FA-10000304","Advantus Push Pins"),
                    ("OFF-FA-10000621","OIC Colored Binder Clips, Assorted Sizes"),
                    ("OFF-FA-10003283","Acco Suede Finish Lanyard with Split Ring"),
                    ("OFF-FA-10002544","Acme Stainless Steel Scissors")],
    "Supplies":    [("OFF-SU-10001218","Fiskars Softgrip Scissors"),
                    ("OFF-SU-10002189","Acme Rosewood Handle Letter Opener"),
                    ("OFF-SU-10004590","Acco Pressboard Data Binder with Storage Hooks"),
                    ("OFF-SU-10003703","Avery Hi-Liter EverBold Pen-Style Highlighter")],
    "Phones":      [("TEC-PH-10002275","Mitel 5320 IP Phone VoIP phone"),
                    ("TEC-PH-10002033","Konftel 250 Conference phone - Charcoal black"),
                    ("TEC-PH-10004977","Apple iPhone 6S"),
                    ("TEC-PH-10001354","Samsung Galaxy S7")],
    "Accessories": [("TEC-AC-10003027","Imation 8GB Mini TravelDrive USB 2.0 Flash Drive"),
                    ("TEC-AC-10000171","Verbatim 25 GB 6x Blu-ray Recordable Disc"),
                    ("TEC-AC-10001814","Plantronics Savi W720 Wireless Headset System"),
                    ("TEC-AC-10002167","Logitech G910 Orion Spark Mechanical Keyboard")],
    "Machines":    [("TEC-MA-10000822","Lexmark MX611dhe Monochrome Laser Printer"),
                    ("TEC-MA-10000864","Cisco 9971 IP Video Phone Charcoal"),
                    ("TEC-MA-10002412","Canon imageCLASS MF4770n Wireless Laser"),
                    ("TEC-MA-10004125","Motorola 56001 Telephone with Digital Answering")],
    "Copiers":     [("TEC-CO-10001449","Hewlett Packard LaserJet 3310 Copier"),
                    ("TEC-CO-10002313","Canon PC1080F Personal Copier"),
                    ("TEC-CO-10004182","Okidata MC860 MFP Laser Printer"),
                    ("TEC-CO-10003723","Sharp AL-1530CS Digital Copier")],
}

PRICE_PROFILE = {
    "Bookcases":   (50,  800,  0.15), "Chairs":      (80,  1500, 0.18),
    "Tables":      (100, 2000, 0.05), "Furnishings": (10,  300,  0.22),
    "Labels":      (5,   60,   0.30), "Storage":     (20,  400,  0.22),
    "Art":         (5,   150,  0.35), "Binders":     (8,   350,  0.28),
    "Appliances":  (30,  800,  0.20), "Paper":       (5,   100,  0.30),
    "Envelopes":   (5,   80,   0.35), "Fasteners":   (5,   50,   0.38),
    "Supplies":    (5,   80,   0.25), "Phones":      (100, 2000, 0.22),
    "Accessories": (15,  500,  0.25), "Machines":    (200, 5000, 0.12),
    "Copiers":     (500, 10000,0.25),
}

SEGMENTS     = ["Consumer","Corporate","Home Office"]
SEG_WEIGHTS  = [0.52, 0.30, 0.18]
SHIP_MODES   = ["Standard Class","Second Class","First Class","Same Day"]
SHIP_WEIGHTS = [0.60, 0.19, 0.15, 0.06]
SHIP_DAYS    = {"Standard Class":(5,7),"Second Class":(3,5),"First Class":(2,3),"Same Day":(0,1)}
DISCOUNTS    = [0.0,0.0,0.0,0.10,0.15,0.20,0.30,0.40,0.45,0.50,0.60,0.70,0.80]

# ================================================================
# SECTION 2 — CUSTOMER POOL
# ================================================================

FIRST_NAMES = ["James","John","Robert","Michael","William","David","Richard","Joseph",
               "Thomas","Charles","Linda","Mary","Patricia","Barbara","Jennifer","Maria",
               "Susan","Margaret","Dorothy","Lisa","Karen","Emily","Daniel","Claire",
               "Mark","Paul","Anna","Sandra","Ashley","Kimberly","Amanda","Jessica",
               "Stephanie","Nicole","Elizabeth","Helen","Sharon","Deborah","Cynthia",
               "Kathleen","Amy","Shirley","Angela","Melissa","Brenda","Gloria","Emma",
               "Ryan","Kevin","Brian","George","Edward","Ronald","Timothy","Jason",
               "Jeffrey","Gary","Larry","Frank","Scott","Eric","Stephen","Andrew"]
LAST_NAMES  = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis",
               "Wilson","Martinez","Anderson","Taylor","Thomas","Jackson","White","Harris",
               "Martin","Thompson","Young","Lewis","Lee","Walker","Hall","Allen","King",
               "Scott","Green","Adams","Nelson","Hill","Baker","Rivera","Campbell","Mitchell",
               "Carter","Roberts","Turner","Phillips","Evans","Torres","Parker","Collins",
               "Edwards","Stewart","Morris","Morgan","Reed","Cook","Bell","Murphy"]

used_names = set()
customers  = []
for i in range(793):
    while True:
        fn = np.random.choice(FIRST_NAMES)
        ln = np.random.choice(LAST_NAMES)
        name = f"{fn} {ln}"
        if name not in used_names:
            used_names.add(name)
            customers.append({"id": f"{fn[0]}{ln[0]}-{10000+i}", "name": name})
            break

ALL_STATES    = list(STATE_REGION.keys())
HIGH_POP      = {"California","Texas","New York","Florida","Illinois","Pennsylvania",
                 "Ohio","Georgia","North Carolina","Michigan","Washington"}
STATE_WEIGHTS = np.array([5.0 if s in HIGH_POP else 1.0 for s in ALL_STATES])
STATE_WEIGHTS /= STATE_WEIGHTS.sum()

# ================================================================
# SECTION 3 — ORDER DATE DISTRIBUTION
# ================================================================

N_ORDERS     = 9994
year_probs   = [0.18, 0.22, 0.28, 0.32]
years_chosen = np.random.choice([2014,2015,2016,2017], size=N_ORDERS, p=year_probs)

order_dates = []
for yr in years_chosen:
    month_w = np.array([0.04,0.04,0.07,0.07,0.07,0.07,0.07,0.08,0.10,0.09,0.12,0.12])
    month_w /= month_w.sum()
    mon      = np.random.choice(range(1,13), p=month_w)
    max_day  = [31,28,31,30,31,30,31,31,30,31,30,31][mon-1]
    if mon == 2 and yr % 4 == 0:
        max_day = 29
    order_dates.append(pd.Timestamp(yr, mon, np.random.randint(1, max_day+1)))

order_dates = sorted(order_dates)

# ================================================================
# SECTION 4 — BUILD ROWS
# ================================================================

print(f"\n[1] Generating {N_ORDERS:,} synthetic orders...")

rows = []
order_id_counter = {}

for i, odate in enumerate(order_dates):
    yr_code = str(odate.year)
    key     = (yr_code, i // 3)
    if key not in order_id_counter:
        prefix = np.random.choice(["CA","US","AA"])
        order_id_counter[key] = f"{prefix}-{yr_code}-{100000+len(order_id_counter)}"
    order_id = order_id_counter[key]

    cust      = customers[np.random.randint(0, len(customers))]
    state     = np.random.choice(ALL_STATES, p=STATE_WEIGHTS)
    city      = np.random.choice(STATE_CITIES[state])
    segment   = np.random.choice(SEGMENTS, p=SEG_WEIGHTS)
    ship_mode = np.random.choice(SHIP_MODES, p=SHIP_WEIGHTS)
    lo, hi    = SHIP_DAYS[ship_mode]
    sdate     = odate + pd.Timedelta(days=int(np.random.randint(lo, hi+1)))

    cat_weights = np.array([0.21, 0.60, 0.19])
    category    = np.random.choice(list(CAT_SUBCAT.keys()), p=cat_weights)
    sub_cat     = np.random.choice(CAT_SUBCAT[category])
    prod_id, prod_name = PRODUCTS[sub_cat][np.random.randint(0, len(PRODUCTS[sub_cat]))]

    pmin, pmax, base_margin = PRICE_PROFILE[sub_cat]
    unit_price = np.random.uniform(pmin, pmax)

    qty_probs  = np.array([0.35,0.20,0.15,0.10,0.07,0.04,0.03,0.02,0.01,0.01,0.01,0.005,0.005,0.005])
    qty_probs /= qty_probs.sum()
    quantity   = int(np.random.choice(range(1, 15), p=qty_probs))

    disc_probs  = np.array([0.40,0.15,0.10,0.08,0.08,0.07,0.05,0.03,0.01,0.01,0.01,0.005,0.005])
    disc_probs /= disc_probs.sum()
    discount    = np.random.choice(DISCOUNTS, p=disc_probs)

    # Stronger seasonal signal
    season_mult = 1.0
    if   odate.month in [11,12]: season_mult = 1.40
    elif odate.month == 9:       season_mult = 1.20
    elif odate.month == 10:      season_mult = 1.10
    elif odate.month in [6,7]:   season_mult = 1.05
    elif odate.month in [1,2]:   season_mult = 0.80

    year_growth = 1.0 + (odate.year - 2014) * 0.08
    sales  = round(unit_price * quantity * (1-discount) * season_mult * year_growth, 2)
    profit = round(sales * (base_margin - discount*0.6 + np.random.normal(0,0.03)), 4)

    rows.append({
        "Row ID": i+1, "Order ID": order_id,
        "Order Date": odate.strftime("%m/%d/%Y"),
        "Ship Date":  sdate.strftime("%m/%d/%Y"),
        "Ship Mode": ship_mode, "Customer ID": cust["id"],
        "Customer Name": cust["name"], "Segment": segment,
        "Country": "United States", "City": city, "State": state,
        "Postal Code": int(np.random.randint(10000,99999)),
        "Region": STATE_REGION[state], "Product ID": prod_id,
        "Category": category, "Sub-Category": sub_cat,
        "Product Name": prod_name, "Sales": sales,
        "Quantity": quantity, "Discount": discount, "Profit": profit,
    })

df_raw = pd.DataFrame(rows)
df_raw.to_csv("Sample - Superstore.csv", index=False)
print(f"✓ Saved 'Sample - Superstore.csv'  ({len(df_raw):,} rows × {len(df_raw.columns)} columns)")

# ================================================================
# SECTION 5 — WEEKLY AGGREGATION
# ================================================================

print("\n[2] Aggregating to weekly sales...")

df_raw["Order Date"] = pd.to_datetime(df_raw["Order Date"])
weekly = (
    df_raw.groupby(pd.Grouper(key="Order Date", freq="W"))["Sales"]
    .sum().reset_index()
    .rename(columns={"Order Date": "date", "Sales": "sales"})
)
weekly.to_csv("superstore_weekly_sales.csv", index=False)
print(f"✓ Saved 'superstore_weekly_sales.csv'  ({len(weekly)} weeks)")

# ================================================================
# SECTION 6 — ENHANCED FEATURE ENGINEERING (Boosted R²)
# ================================================================

print("\n[3] Engineering features (enhanced)...")

w = weekly.copy()
w["date"] = pd.to_datetime(w["date"])
w.sort_values("date", inplace=True)
w.reset_index(drop=True, inplace=True)

# Calendar features
w["week"]        = w["date"].dt.isocalendar().week.astype(int)
w["month"]       = w["date"].dt.month
w["quarter"]     = w["date"].dt.quarter
w["year"]        = w["date"].dt.year
w["day_of_year"] = w["date"].dt.dayofyear

# Cyclical encoding
w["sin_month"] = np.sin(2 * np.pi * w["month"] / 12)
w["cos_month"] = np.cos(2 * np.pi * w["month"] / 12)
w["sin_week"]  = np.sin(2 * np.pi * w["week"] / 52)
w["cos_week"]  = np.cos(2 * np.pi * w["week"] / 52)
w["sin_qtr"]   = np.sin(2 * np.pi * w["quarter"] / 4)
w["cos_qtr"]   = np.cos(2 * np.pi * w["quarter"] / 4)

# Season flags
w["is_holiday_season"] = w["month"].isin([11,12]).astype(int)
w["is_q4"]     = (w["quarter"] == 4).astype(int)
w["is_q1"]     = (w["quarter"] == 1).astype(int)
w["is_summer"] = w["month"].isin([6,7,8]).astype(int)
w["is_sep_oct"]= w["month"].isin([9,10]).astype(int)  # pre-holiday ramp

# Year trend (normalised)
w["year_trend"]       = (w["year"] - w["year"].min()) / max(1, (w["year"].max() - w["year"].min()))
w["year_trend_sq"]    = w["year_trend"] ** 2  # capture acceleration

# ── NEW: Fourier terms for annual seasonality (stronger signal)
for k in [1, 2, 3]:
    w[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * w["day_of_year"] / 365.25)
    w[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * w["day_of_year"] / 365.25)

# Lag features
for lag in [1, 2, 3, 4, 6, 8, 12, 16, 26, 52]:
    w[f"lag_{lag}"] = w["sales"].shift(lag)

# Rolling statistics
for window in [4, 8, 12, 26, 52]:
    w[f"rolling_{window}"]     = w["sales"].shift(1).rolling(window).mean()
    w[f"rolling_std_{window}"] = w["sales"].shift(1).rolling(window).std()

# Exponential weighted moving average (better trend capture)
for span in [4, 8, 12]:
    w[f"ewm_{span}"] = w["sales"].shift(1).ewm(span=span).mean()

# Momentum
w["momentum_4"] = w["sales"].shift(1) - w["sales"].shift(5)
w["momentum_8"] = w["sales"].shift(1) - w["sales"].shift(9)

# ── NEW: Year-over-year ratio (very powerful for seasonality)
w["yoy_ratio"] = w["sales"].shift(52) / (w["sales"].shift(53) + 1e-9)

w.dropna(inplace=True)
w.reset_index(drop=True, inplace=True)

w.to_csv("superstore_weekly_features.csv", index=False)
print(f"✓ Saved 'superstore_weekly_features.csv'  ({len(w)} weeks, {len(w.columns)} features)")

# ================================================================
# SECTION 7 — CLEAN EDA DASHBOARD (Fixed Layout & Labels)
# ================================================================

print("\n[4] Generating EDA dashboard...")

fig = plt.figure(figsize=(20, 13))
fig.patch.set_facecolor('#0b0f1a')
fig.suptitle(
    "Superstore Sales  —  EDA Dashboard  |  Future Interns ML Task 1 (2026)",
    color="white", fontsize=14, fontweight="bold", y=0.99
)

def style_ax(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor("#111827")
    ax.set_title(title, color="white", fontsize=9, pad=8, fontweight="bold")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1e2d45")
    ax.grid(color="#1e2d45", linewidth=0.5, alpha=0.5, axis='both')
    if xlabel: ax.set_xlabel(xlabel, color="#94a3b8", fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color="#94a3b8", fontsize=8)

gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.38,
                      left=0.06, right=0.97, top=0.94, bottom=0.06)

# ── Plot 1: Weekly Sales Timeline (full width)
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(weekly["date"], weekly["sales"], color="#3b82f6", linewidth=1.2, alpha=0.85)
ax1.axhline(weekly["sales"].mean(), color="#ef4444", linestyle="--",
            linewidth=1.2, label=f"Mean  ${weekly['sales'].mean():,.0f}")
ax1.set_ylabel("Weekly Sales ($)", color="#94a3b8", fontsize=8)
ax1.legend(loc='upper left', facecolor="#1e2d45", labelcolor="white",
           edgecolor="#334155", fontsize=8, framealpha=0.9)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
style_ax(ax1, "Weekly Sales Timeline  (2014 – 2017)")

# ── Plot 2: Sales by Category
ax2 = fig.add_subplot(gs[1, 0])
cat_sales = df_raw.groupby("Category")["Sales"].sum().sort_values()
bars = ax2.barh(cat_sales.index, cat_sales.values,
                color=["#3b82f6","#10b981","#f59e0b"])
for bar, val in zip(bars, cat_sales.values):
    ax2.text(val * 1.02, bar.get_y() + bar.get_height()/2,
             f"${val/1e6:.2f}M", va="center", color="white", fontsize=7.5)
ax2.set_xlim(0, cat_sales.max() * 1.35)
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
style_ax(ax2, "Sales by Category", xlabel="Total Sales")

# ── Plot 3: Sales by Region (Pie)
ax3 = fig.add_subplot(gs[1, 1])
region_sales = df_raw.groupby("Region")["Sales"].sum()
wedges, texts, autotexts = ax3.pie(
    region_sales, labels=region_sales.index, autopct="%1.1f%%",
    colors=["#3b82f6","#10b981","#f59e0b","#8b5cf6"],
    textprops={"color":"white","fontsize":7.5},
    pctdistance=0.75, startangle=90
)
for at in autotexts:
    at.set_color("white"); at.set_fontsize(7)
style_ax(ax3, "Sales by Region")
ax3.grid(False)

# ── Plot 4: Sales by Segment
ax4 = fig.add_subplot(gs[1, 2])
seg_sales = df_raw.groupby("Segment")["Sales"].sum().sort_values()
bars4 = ax4.barh(seg_sales.index, seg_sales.values,
                 color=["#8b5cf6","#06b6d4","#10b981"])
for bar, val in zip(bars4, seg_sales.values):
    ax4.text(val * 1.02, bar.get_y() + bar.get_height()/2,
             f"${val/1e6:.2f}M", va="center", color="white", fontsize=7.5)
ax4.set_xlim(0, seg_sales.max() * 1.38)
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
style_ax(ax4, "Sales by Segment", xlabel="Total Sales")

# ── Plot 5: Monthly Avg Sales (Seasonality)
ax5 = fig.add_subplot(gs[1, 3])
df_raw["month_num"] = df_raw["Order Date"].dt.month
monthly_avg = df_raw.groupby("month_num")["Sales"].mean()
month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
ax5.bar(range(1,13), monthly_avg.values, color="#f59e0b", alpha=0.85)
ax5.set_xticks(range(1,13)); ax5.set_xticklabels(month_labels, fontsize=7, rotation=45)
ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
style_ax(ax5, "Avg Sales by Month (Seasonality)", ylabel="Avg Sales")

# ── Plot 6: Top 10 Sub-Categories
ax6 = fig.add_subplot(gs[2, :2])
top10 = df_raw.groupby("Sub-Category")["Sales"].sum().nlargest(10).sort_values()
bars6 = ax6.barh(top10.index, top10.values, color="#f59e0b", alpha=0.9)
for bar, val in zip(bars6, top10.values):
    ax6.text(val * 1.01, bar.get_y() + bar.get_height()/2,
             f"${val/1e3:.0f}K", va="center", color="white", fontsize=7.5)
ax6.set_xlim(0, top10.max() * 1.25)
ax6.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e3:.0f}K"))
style_ax(ax6, "Top 10 Sub-Categories by Revenue", xlabel="Total Sales")

# ── Plot 7: Profit Margin by Category
ax7 = fig.add_subplot(gs[2, 2])
df_raw["margin_pct"] = df_raw["Profit"] / df_raw["Sales"].replace(0, np.nan) * 100
margin = df_raw.groupby("Category")["margin_pct"].mean().sort_values()
bars7 = ax7.barh(margin.index, margin.values,
                 color=["#ef4444","#10b981","#3b82f6"])
for bar, val in zip(bars7, margin.values):
    ax7.text(val + 0.3, bar.get_y() + bar.get_height()/2,
             f"{val:.1f}%", va="center", color="white", fontsize=7.5)
ax7.set_xlim(0, margin.max() * 1.35)
ax7.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0f}%"))
style_ax(ax7, "Profit Margin by Category", xlabel="Avg Margin (%)")

# ── Plot 8: Yearly Sales Growth
ax8 = fig.add_subplot(gs[2, 3])
df_raw["year_num"] = df_raw["Order Date"].dt.year
yearly = df_raw.groupby("year_num")["Sales"].sum()
bars8 = ax8.bar(yearly.index, yearly.values, color=["#3b82f6","#06b6d4","#10b981","#f59e0b"])
for bar, val in zip(bars8, yearly.values):
    ax8.text(bar.get_x() + bar.get_width()/2, val * 1.01,
             f"${val/1e6:.2f}M", ha="center", va="bottom", color="white", fontsize=7.5)
ax8.set_xlim(2013.5, 2017.5)
ax8.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax8.set_xticks([2014,2015,2016,2017])
style_ax(ax8, "Year-over-Year Sales Growth", ylabel="Total Sales")

plt.savefig("superstore_eda.png", dpi=150, facecolor=fig.get_facecolor(),
            bbox_inches='tight')
print("✓ Saved 'superstore_eda.png'")
plt.close()

print("\n" + "=" * 60)
print("  data.py complete!  Next step: python model.py")
print("=" * 60)
