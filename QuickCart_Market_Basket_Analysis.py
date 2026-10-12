#!/usr/bin/env python
# coding: utf-8

# # QuickCart Market Basket Analysis
# ### Apriori & FP-Growth on 5,000 real baskets — Unsupervised ML Minor Project
# 
# **Business question:** which products get bought **together**? We mine association rules of the form
# *"if a basket contains A, it is likely to also contain B"* and turn the strongest ones into
# **shelf placement**, **bundle deals** and **"frequently bought together"** app recommendations.
# 
# | Metric | Formula | Question it answers |
# |---|---|---|
# | Support | P(A ∧ B) | How common is the combination overall? |
# | Confidence | P(B \| A) | Given A was bought, how likely is B? |
# | Lift | Confidence / P(B) | Is the combination more common than chance? (>1 = yes) |
# 
# **Notebook outline**
# 1. Setup & load data
# 2. Sanity checks against the spec's locked numbers
# 3. Exploratory analysis (basket sizes, top SKUs, categories, stores)
# 4. Reshape: long → basket list → one-hot matrix
# 5. Mine frequent itemsets with Apriori **and** FP-Growth (+ runtime benchmark)
# 6. Generate association rules
# 7. Verify the 6 planted patterns (answer key)
# 8. The "high lift, low support" trap (compound rules)
# 9. Control check: Banana → Shampoo
# 10. Actionable rules
# 11. Business recommendations: shelf placement, bundles, app cross-sell
# 12. Validation against the reference answer-key files
# 13. Conclusions

# ## 1. Setup & load data

# In[1]:


import time, warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 160)
sns.set_theme(style="whitegrid", palette="deep")

DATA = Path("data")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
CHARTS = Path("charts"); CHARTS.mkdir(exist_ok=True)

MIN_SUPPORT = 0.02   # spec: min_support = 0.02
MIN_LIFT = 1.0       # spec: rules with lift >= 1.0

tx = pd.read_csv(DATA / "quickcart_basket_transactions.csv", parse_dates=["date"])
skus = pd.read_csv(DATA / "dim_skus.csv")
print(tx.shape, skus.shape)
tx.head()


# In[2]:


skus.head()


# ## 2. Sanity checks against the spec's locked numbers
# Every check below is an `assert`, so the notebook fails loudly if the data does not match the spec.

# In[3]:


basket_size = tx.groupby("order_id")["sku_id"].nunique()

checks = {
    "line items":            (len(tx), 26_878),
    "unique orders":         (tx["order_id"].nunique(), 5_000),
    "unique SKUs":           (tx["sku_id"].nunique(), 60),
    "stores":                (tx["store_id"].nunique(), 12),
    "first date":            (str(tx["date"].min().date()), "2026-11-01"),
    "last date":             (str(tx["date"].max().date()), "2026-11-30"),
    "mean basket size":      (round(basket_size.mean(), 2), 5.38),
    "median basket size":    (basket_size.median(), 5),
    "min basket size":       (basket_size.min(), 1),
    "max basket size":       (basket_size.max(), 16),
}
res = pd.DataFrame(checks, index=["actual", "spec"]).T
res["match"] = res["actual"] == res["spec"]
display(res)
assert res["match"].all()

# data-quality checks
print("Missing values:", tx.isna().sum().sum())
print("Duplicate (order, SKU) lines:", tx.duplicated(["order_id", "sku_id"]).sum())
print("Orders spanning >1 date/store:",
      (tx.groupby("order_id")[["date", "store_id"]].nunique() > 1).any(axis=1).sum())
print("SKU ids not in catalog:", (~tx["sku_id"].isin(skus["sku_id"])).sum())
# denormalised names/categories agree with the catalog
chk = tx[["sku_id", "sku_name", "category"]].drop_duplicates().merge(skus, on="sku_id", suffixes=("", "_dim"))
assert (chk["sku_name"] == chk["sku_name_dim"]).all() and (chk["category"] == chk["category_dim"]).all()
print("sku_name / category consistent with dim_skus.csv ✔")


# ## 3. Exploratory analysis

# In[4]:


fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
basket_size.value_counts().sort_index().plot.bar(ax=axes[0], color="#4C72B0")
axes[0].axvline(basket_size.mean() - 1, color="crimson", ls="--", label=f"mean = {basket_size.mean():.2f}")
axes[0].set(title="Basket size distribution", xlabel="Items per basket", ylabel="Orders"); axes[0].legend()

item_support = (tx.groupby("sku_name")["order_id"].nunique() / tx["order_id"].nunique()).sort_values(ascending=False)
item_support.head(15).sort_values().plot.barh(ax=axes[1], color="#55A868")
axes[1].set(title="Top 15 SKUs by single-item support", xlabel="Support (share of baskets)")
axes[1].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
plt.tight_layout(); plt.savefig(CHARTS / "01_basket_size_and_top_skus.png", dpi=150); plt.show()


# In[5]:


top5_spec = {"White Bread 400g": 0.2546, "Toned Milk 1L": 0.2540, "Potato Chips 90g": 0.2184,
             "Cola 750ml": 0.2164, "Toor Dal 1kg": 0.2068}
top5 = item_support.head(5).round(4)
display(pd.DataFrame({"actual": top5, "spec": pd.Series(top5_spec)}))
assert top5.to_dict() == top5_spec, "Top-5 single-item support does not match the spec"
print("Top-5 single-item support matches the spec ✔")


# In[6]:


fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
cat = tx.groupby("category")["order_id"].nunique().div(5000).sort_values()
cat.plot.barh(ax=axes[0], color="#8172B2"); axes[0].set(title="Share of baskets containing each category", xlabel="")
axes[0].xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
tx.groupby("store_id")["order_id"].nunique().plot.bar(ax=axes[1], color="#C44E52")
axes[1].set(title="Orders per store", xlabel="", ylabel="Orders")
tx.groupby(tx["date"].dt.date)["order_id"].nunique().plot(ax=axes[2], marker="o", color="#DD8452")
axes[2].set(title="Orders per day (Nov 2026)", xlabel="", ylabel="Orders"); axes[2].tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.savefig(CHARTS / "02_categories_stores_days.png", dpi=150); plt.show()


# ## 4. Reshape: long → basket list → one-hot
# The CSV has **one row per line item**. Apriori / FP-Growth need **one row per basket**, so the first real task is a reshape:
# 
# `long (26,878 rows)` → `list of items per order (5,000 lists)` → `one-hot matrix (5,000 × 60, True/False)`

# In[7]:


baskets = tx.groupby("order_id")["sku_name"].apply(list)
print("Example baskets:")
for oid, items in baskets.head(3).items():
    print(f"  {oid}: {items}")

te = TransactionEncoder()
te_ary = te.fit(baskets.tolist()).transform(baskets.tolist())
df_encoded = pd.DataFrame(te_ary, columns=te.columns_, index=baskets.index)
print("\nOne-hot matrix shape:", df_encoded.shape)
assert df_encoded.shape == (5000, 60)
assert df_encoded.sum().sum() == len(tx)   # every line item became exactly one True
df_encoded.iloc[:5, :8]


# ## 5. Mine frequent itemsets — Apriori vs FP-Growth

# In[8]:


t0 = time.perf_counter()
frequent_itemsets = apriori(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
t_ap = time.perf_counter() - t0

t0 = time.perf_counter()
frequent_itemsets_fp = fpgrowth(df_encoded, min_support=MIN_SUPPORT, use_colnames=True)
t_fp = time.perf_counter() - t0

frequent_itemsets["length"] = frequent_itemsets["itemsets"].apply(len)
frequent_itemsets_fp["length"] = frequent_itemsets_fp["itemsets"].apply(len)

# Same result? compare as {itemset: support}
ap_map = dict(zip(frequent_itemsets["itemsets"], frequent_itemsets["support"].round(10)))
fp_map = dict(zip(frequent_itemsets_fp["itemsets"], frequent_itemsets_fp["support"].round(10)))
assert ap_map == fp_map
print(f"Apriori:   {len(frequent_itemsets)} itemsets in {t_ap*1000:.1f} ms")
print(f"FP-Growth: {len(frequent_itemsets_fp)} itemsets in {t_fp*1000:.1f} ms")
print("Identical itemsets and supports ✔")

counts = frequent_itemsets["length"].value_counts().sort_index()
spec_counts = pd.Series({1: 60, 2: 78, 3: 86, 4: 30, 5: 2})
display(pd.DataFrame({"actual": counts, "spec": spec_counts}).rename_axis("itemset size"))
assert counts.equals(spec_counts) and len(frequent_itemsets) == 256
print("Itemset counts match the spec (256 total) ✔")


# ### Runtime benchmark across support thresholds
# Lower support = more candidate itemsets = where FP-Growth's single-pass tree approach pays off. Each timing is the best of 3 runs.

# In[9]:


def best_time(fn, n=3):
    times = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); times.append(time.perf_counter() - t0)
    return min(times), len(out)

bench = []
for s in [0.05, 0.03, 0.02, 0.015, 0.01, 0.0075, 0.005, 0.003, 0.002]:
    ta, na = best_time(lambda: apriori(df_encoded, min_support=s, use_colnames=True))
    tf, nf = best_time(lambda: fpgrowth(df_encoded, min_support=s, use_colnames=True))
    assert na == nf
    bench.append({"min_support": s, "itemsets": na, "apriori_ms": ta*1000, "fpgrowth_ms": tf*1000})
bench = pd.DataFrame(bench)
bench["speedup_fp_vs_apriori"] = bench["apriori_ms"] / bench["fpgrowth_ms"]
display(bench.style.format({"min_support": "{:.4f}", "apriori_ms": "{:.1f}", "fpgrowth_ms": "{:.1f}", "speedup_fp_vs_apriori": "{:.2f}x"}))
bench.to_csv(OUT / "benchmark_apriori_vs_fpgrowth.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(bench["min_support"], bench["apriori_ms"], marker="o", label="Apriori")
ax.plot(bench["min_support"], bench["fpgrowth_ms"], marker="s", label="FP-Growth")
ax.invert_xaxis(); ax.set_yscale("log")
for _, r in bench.iterrows():
    ax.annotate(f"{int(r.itemsets)}", (r.min_support, r.apriori_ms), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.set(title="Runtime vs min_support (labels = # frequent itemsets)", xlabel="min_support (lower →)", ylabel="Runtime (ms, log)")
ax.legend(); plt.tight_layout(); plt.savefig(CHARTS / "03_benchmark.png", dpi=150); plt.show()


# **Takeaway:** both algorithms return *exactly* the same itemsets and supports — they differ only in how they search.
# On this small 60-SKU catalog, at the spec's `min_support = 0.02`, **mlxtend's Apriori is actually faster**: with few candidates,
# its vectorised NumPy candidate counting beats the overhead of building an FP-tree. The picture flips as `min_support` drops and the
# number of frequent itemsets climbs into the thousands: Apriori's candidate generation blows up, while FP-Growth's runtime stays nearly flat,
# so it ends up roughly 2-3x faster by min_support = 0.002 (last rows of the table). For a real QuickCart catalog with thousands of SKUs and
# lower thresholds, FP-Growth is the practical default; for a 60-SKU teaching dataset either is fine.

# ## 6. Generate association rules

# In[10]:


rules = association_rules(frequent_itemsets.drop(columns="length"), num_itemsets=len(df_encoded),
                          metric="lift", min_threshold=MIN_LIFT)
rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
print("Rules generated:", len(rules))
assert len(rules) == 980
print("Rule count matches the spec (980) ✔")

fmt = lambda s: ", ".join(sorted(s))
rules["antecedent"] = rules["antecedents"].apply(fmt)
rules["consequent"] = rules["consequents"].apply(fmt)
rules["n_antecedent"] = rules["antecedents"].apply(len)
rules["n_consequent"] = rules["consequents"].apply(len)
rules["support_baskets"] = (rules["support"] * len(df_encoded)).round().astype(int)

COLS = ["antecedent", "consequent", "support", "support_baskets", "confidence", "lift", "leverage", "conviction"]
rules[COLS].head(10).round(4)


# In[11]:


rules_out = rules[["antecedent", "consequent", "n_antecedent", "n_consequent", "antecedent support",
                   "consequent support", "support", "support_baskets", "confidence", "lift",
                   "leverage", "conviction", "zhangs_metric", "jaccard"]]
rules_out.round(6).to_csv(OUT / "association_rules_all.csv", index=False)
fi_out = frequent_itemsets.assign(itemset=frequent_itemsets["itemsets"].apply(fmt))[["itemset", "length", "support"]]
fi_out.sort_values(["length", "support"], ascending=[True, False]).to_csv(OUT / "frequent_itemsets_apriori.csv", index=False)
print("Saved outputs/association_rules_all.csv and outputs/frequent_itemsets_apriori.csv")


# ## 7. Verify the 6 planted patterns (locked answer key)

# In[12]:


planted = pd.DataFrame([
    ("Breakfast Combo",       "Toned Milk 1L",        "White Bread 400g", 0.2384, 0.9386, 3.69),
    ("Thali Staples",         "Basmati Rice 5kg",     "Toor Dal 1kg",     0.1534, 0.7740, 3.74),
    ("Snack-Drink Combo",     "Potato Chips 90g",     "Cola 750ml",       0.1974, 0.9038, 4.18),
    ("Ghar Breakfast",        "Ghee 500ml",           "Poha 500g",        0.1468, 0.8605, 5.07),
    ("Tea-Time",              "Instant Coffee 100g",  "Choco Cookies 150g", 0.1520, 0.8588, 4.84),
    ("Personal Care Restock", "Hand Sanitizer 200ml", "Face Wash 100ml",  0.1318, 0.7930, 5.05),
], columns=["pattern", "antecedent", "consequent", "spec_support", "spec_confidence", "spec_lift"])

pv = planted.merge(rules[["antecedent", "consequent", "support", "confidence", "lift", "support_baskets"]],
                   on=["antecedent", "consequent"], how="left")
pv["support_ok"] = (pv["support"].round(4) == pv["spec_support"])
pv["confidence_ok"] = (pv["confidence"].round(4) == pv["spec_confidence"])
pv["lift_ok"] = (pv["lift"].round(2) == pv["spec_lift"])
display(pv.round(4))
assert pv[["support_ok", "confidence_ok", "lift_ok"]].all().all()
print("All 6 planted patterns reproduced exactly ✔")
pv.round(4).to_csv(OUT / "planted_patterns_verification.csv", index=False)


# In[13]:


# The reverse direction of each planted rule - confidence is asymmetric, lift is not.
rev = planted.rename(columns={"antecedent": "consequent", "consequent": "antecedent"})[["pattern", "antecedent", "consequent"]]
rev = rev.merge(rules[["antecedent", "consequent", "support", "confidence", "lift"]], on=["antecedent", "consequent"])
print("Reverse direction (B → A):")
rev.round(4)


# Lift is symmetric (A→B and B→A have the same lift), but **confidence is not**. E.g. `Toned Milk → White Bread` has
# ~94% confidence while `White Bread → Toned Milk` is lower, because bread is also bought without milk.
# This matters for app recommendations: the trigger item should be the **antecedent of the higher-confidence direction**.

# ## 8. The trap: very high lift on very low support

# In[14]:


print("Top 10 rules by lift:")
display(rules[COLS + ["n_antecedent", "n_consequent"]].head(10).round(4))

top = rules.iloc[0]
print(f"Highest-lift rule: {{{top.antecedent}}} → {{{top.consequent}}}  "
      f"lift = {top.lift:.2f}, support = {top.support:.2%} ({top.support_baskets} baskets)")
assert 26.5 < top.lift < 27.0 and top.n_antecedent == 2 and top.n_consequent == 2


# In[15]:


planted_keys = set(zip(planted.antecedent, planted.consequent)) | set(zip(planted.consequent, planted.antecedent))
rules["is_planted_pair"] = [(a, c) in planted_keys for a, c in zip(rules.antecedent, rules.consequent)]
rules["rule_size"] = rules["n_antecedent"] + rules["n_consequent"]

fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(rules["support"], rules["lift"], c=rules["rule_size"], cmap="viridis", s=18, alpha=0.6)
p = rules[rules.is_planted_pair]
ax.scatter(p["support"], p["lift"], s=120, facecolors="none", edgecolors="crimson", linewidths=2, label="Planted pairs (both directions)")
ax.axvline(0.05, color="grey", ls="--"); ax.axhline(2, color="grey", ls="--")
ax.fill_between([0.05, rules.support.max()*1.05], 2, rules.lift.max()*1.05, color="green", alpha=0.05)
ax.text(0.12, 9, "actionable zone: support > 5% and lift > 2", fontsize=9, color="darkgreen")
t = rules.iloc[0]
ax.annotate("compound 2→2 rules:\nlift ≈ 27 but only ~2% support", (t.support, t.lift), xytext=(0.07, 22), textcoords="data",
            arrowprops=dict(arrowstyle="->", color="black", lw=1, shrinkA=0, shrinkB=4), fontsize=9)
plt.colorbar(sc, label="Items in rule")
ax.set(title="All 980 rules: lift vs support", xlabel="Support", ylabel="Lift")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
ax.legend(loc="upper right"); plt.tight_layout(); plt.savefig(CHARTS / "04_lift_vs_support.png", dpi=150); plt.show()


# In[16]:


summary = rules.groupby("rule_size").agg(rules=("lift", "size"), median_support=("support", "median"),
                                         median_lift=("lift", "median"), max_lift=("lift", "max")).round(4)
summary


# **Why the highest-lift rules are not the best rules.** The ~26.8-lift rules (e.g. `{Hand Sanitizer, Instant Coffee} → {Face Wash, Choco Cookies}`)
# appear when **two planted patterns happen to land in the same basket** (Tea-Time + Personal Care Restock). Their lift is huge because the
# combined consequent is rare (~2.7% of baskets), but they rest on only ~100 baskets out of 5,000. A planted pair such as
# `Toned Milk → White Bread` has a "modest" lift of 3.69 but is backed by ~1,190 baskets — it is far more reliable and touches far more customers.
# 
# > **Rule of thumb:** never rank on lift alone. Require reasonable support *and* lift, and prefer simple 1→1 rules for business actions.

# ## 9. Control check: Banana 1kg → Shampoo 340ml
# Two categories with no designed relationship. If the method found patterns in anything, this pair would show up.

# In[17]:


a, b = "Banana 1kg", "Shampoo 340ml"
pA, pB = df_encoded[a].mean(), df_encoded[b].mean()
pAB = (df_encoded[a] & df_encoded[b]).mean()
ctrl = pd.Series({"support(Banana)": pA, "support(Shampoo)": pB, "support(both)": pAB,
                  "baskets with both": int(pAB * len(df_encoded)),
                  "confidence Banana→Shampoo": pAB / pA if pA else np.nan,
                  "lift": pAB / (pA * pB) if pA * pB else np.nan})
display(ctrl.round(4).to_frame("value"))

in_itemsets = frequent_itemsets["itemsets"].apply(lambda s: {a, b} <= s).any()
in_rules = rules.apply(lambda r: {a, b} <= (r.antecedents | r.consequents), axis=1).any()
print(f"Pair in frequent itemsets: {in_itemsets} | Pair in any rule: {in_rules}")
assert pAB < MIN_SUPPORT and not in_itemsets and not in_rules
print("Control pair does NOT clear min_support = 2% → the planted patterns are genuine signal, not mining artefacts ✔")


# In[18]:


# Broader view: support & lift of every SKU pair, split by planted vs everything else
pairs = []
cols = df_encoded.columns
X = df_encoded.values.astype(int)
co = X.T @ X
sup = X.mean(axis=0)
for i, j in combinations(range(len(cols)), 2):
    s_ij = co[i, j] / len(X)
    pairs.append((cols[i], cols[j], s_ij, s_ij / (sup[i] * sup[j]) if s_ij else 0.0))
pairs = pd.DataFrame(pairs, columns=["item_a", "item_b", "support", "lift"])
pairs["planted"] = [(x, y) in planted_keys for x, y in zip(pairs.item_a, pairs.item_b)]
print(f"{len(pairs)} possible SKU pairs | pairs with support >= 2%: {(pairs.support >= MIN_SUPPORT).sum()}")
display(pairs.groupby("planted")[["support", "lift"]].describe().round(3).T)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(pairs.loc[~pairs.planted, "lift"], bins=60, color="#4C72B0", alpha=0.8, label="All other pairs")
for _, r in pairs[pairs.planted].iterrows():
    ax.axvline(r.lift, color="crimson", lw=2)
ax.axvline(np.nan, color="crimson", lw=2, label="Planted pairs")
ax.set(title="Lift of all 1,770 SKU pairs: background noise sits near 1.0", xlabel="Lift", ylabel="Pairs"); ax.legend()
plt.tight_layout(); plt.savefig(CHARTS / "05_pair_lift_distribution.png", dpi=150); plt.show()


# ## 10. Actionable rules
# Spec filter: `support > 0.05` **and** `lift > 2`.

# In[19]:


actionable = rules[(rules["support"] > 0.05) & (rules["lift"] > 2)].copy()
print(f"{len(actionable)} actionable rules out of {len(rules)}")
display(actionable[COLS].round(4))
actionable[["antecedent", "consequent", "support", "support_baskets", "confidence", "lift", "leverage", "conviction"]] \
    .round(4).to_csv(OUT / "actionable_rules.csv", index=False)


# In[20]:


# Which planted pattern does each actionable rule belong to?
pattern_items = {r.pattern: {r.antecedent, r.consequent} for r in planted.itertuples()}
def which(row):
    items = row.antecedents | row.consequents
    hit = [p for p, s in pattern_items.items() if s <= items]
    if not hit:
        return "Thali Staples + Cooking Oil" if "Cooking Oil 1L" in items else "other"
    kind = "core pair" if items in pattern_items.values() else "pair + extra item"
    return f"{' & '.join(hit)} ({kind})"
actionable["pattern"] = actionable.apply(which, axis=1)
display(actionable["pattern"].value_counts().to_frame("rules"))
actionable[actionable.n_antecedent + actionable.n_consequent == 2][["antecedent", "consequent", "support", "confidence", "lift"]].round(4)


# The actionable filter is a strong sieve. Every one of the 42 surviving rules is built on a planted pattern:
# - the **12 directed core rules** of the six planted pairs;
# - **Cooking Oil 1L** turns out to be a hidden third member of the *Thali Staples* pattern (Basmati / Toor Dal / Cooking Oil all pair at lift ≈ 3.7 and ~15% support), something the answer-key table does not list;
# - the remainder are a planted pair plus one very popular item (e.g. `{Cola, Toned Milk} → Potato Chips`), which add little beyond the core pair.
# 
# The compound high-lift rules from section 8 are all filtered out for having support below 5%.

# ## 11. Business recommendations

# ### 11a. Shelf placement — which SKUs belong next to each other in the dark-store picking layout?
# Build a graph where SKUs are nodes and an edge exists for every pair rule with support ≥ 5% and lift ≥ 2; connected components are the 'pick zones'.

# In[21]:


pair_rules = rules[(rules.n_antecedent == 1) & (rules.n_consequent == 1)]
strong = pair_rules[(pair_rules.support >= 0.05) & (pair_rules.lift >= 2)]

# union-find over strong edges
parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for a_, c_ in zip(strong.antecedent, strong.consequent):
    parent[find(a_)] = find(c_)
zones = {}
for n in set(strong.antecedent) | set(strong.consequent):
    zones.setdefault(find(n), set()).add(n)

cat_of = dict(zip(skus.sku_name, skus.category))
zone_rows = []
for i, members in enumerate(sorted(zones.values(), key=lambda m: -max(item_support[x] for x in m)), 1):
    sub = strong[strong.antecedent.isin(members) & strong.consequent.isin(members)]
    zone_rows.append({"zone": f"Zone {i}", "skus": " + ".join(sorted(members)),
                      "categories": " / ".join(sorted({cat_of[m] for m in members})),
                      "max_pair_support": sub.support.max(), "max_lift": sub.lift.max()})
shelf = pd.DataFrame(zone_rows)
shelf["cross_category"] = shelf["categories"].str.contains("/")
display(shelf.round(4))
shelf.round(4).to_csv(OUT / "rec_shelf_placement.csv", index=False)


# In[22]:


# Co-purchase lift heatmap for the 12 SKUs involved in the planted patterns + a few popular non-planted SKUs
focus = sorted({x for s in pattern_items.values() for x in s} | {"Cooking Oil 1L"}) + ["Banana 1kg", "Shampoo 340ml", "Sugar 1kg", "Onion 1kg"]
L = pd.DataFrame(index=focus, columns=focus, dtype=float)
idx = {c: k for k, c in enumerate(cols)}
for x in focus:
    for y in focus:
        L.loc[x, y] = np.nan if x == y else (co[idx[x], idx[y]] / len(X)) / (sup[idx[x]] * sup[idx[y]])
plt.figure(figsize=(11, 9)); plt.grid(False)
sns.heatmap(L, annot=True, fmt=".1f", cmap="RdBu_r", center=1, vmin=0, vmax=5.5, cbar_kws={"label": "Lift"})
plt.title("Pairwise lift: planted pairs light up, everything else ≈ 1.0"); plt.tight_layout()
plt.savefig(CHARTS / "06_lift_heatmap.png", dpi=150); plt.show()


# ### 11b. Bundle deals — which pairs are strong enough to package as a discounted combo?

# In[23]:


price = dict(zip(skus.sku_name, skus.unit_price_inr))
b = planted[["pattern", "antecedent", "consequent"]].merge(
        pair_rules[["antecedent", "consequent", "support", "confidence", "lift"]], on=["antecedent", "consequent"])
# reverse-direction confidence
rc = pair_rules.set_index(["antecedent", "consequent"])["confidence"]
b["reverse_confidence"] = [rc.get((c_, a_), np.nan) for a_, c_ in zip(b.antecedent, b.consequent)]
b["bundle_baskets_nov"] = (b["support"] * len(df_encoded)).round().astype(int)
b["price_a_inr"] = b.antecedent.map(price); b["price_b_inr"] = b.consequent.map(price)
b["bundle_list_price_inr"] = b.price_a_inr + b.price_b_inr
# baskets with A but not B (and B but not A) = the uplift opportunity a bundle discount targets
b["baskets_A_only"] = [int((df_encoded[a_] & ~df_encoded[c_]).sum()) for a_, c_ in zip(b.antecedent, b.consequent)]
b["baskets_B_only"] = [int((df_encoded[c_] & ~df_encoded[a_]).sum()) for a_, c_ in zip(b.antecedent, b.consequent)]

def bundle_call(r):
    if r.support >= 0.15 and min(r.confidence, r.reverse_confidence) >= 0.75:
        return "Everyday combo - small discount (3-5%); already bought together, protect basket"
    if r.lift >= 4.5:
        return "Hero bundle - 8-10% off; strongest affinity, promote on home screen"
    return "Standard bundle - 5-8% off; convert the single-item buyers"
b["recommendation"] = b.apply(bundle_call, axis=1)
# 3-item Thali bundle: Basmati + Toor Dal + Cooking Oil
tri = ["Basmati Rice 5kg", "Toor Dal 1kg", "Cooking Oil 1L"]
tri_sup = df_encoded[tri].all(axis=1).mean()
tri_rule = rules[(rules.antecedent == "Basmati Rice 5kg, Toor Dal 1kg") & (rules.consequent == "Cooking Oil 1L")].iloc[0]
b = pd.concat([b, pd.DataFrame([{
    "pattern": "Thali Staples (3-item)", "antecedent": "Basmati Rice 5kg, Toor Dal 1kg", "consequent": "Cooking Oil 1L",
    "support": tri_sup, "confidence": tri_rule.confidence, "lift": tri_rule.lift, "reverse_confidence": np.nan,
    "bundle_baskets_nov": int(round(tri_sup * len(df_encoded))),
    "price_a_inr": price[tri[0]] + price[tri[1]], "price_b_inr": price[tri[2]],
    "bundle_list_price_inr": sum(price[t] for t in tri),
    "baskets_A_only": int((df_encoded[tri[:2]].all(axis=1) & ~df_encoded[tri[2]]).sum()), "baskets_B_only": np.nan,
    "recommendation": "Monthly-staples kit - 5% off; three-way affinity, ideal for subscription / 'refill' bundle"}])],
    ignore_index=True)
bundles = b.sort_values("lift", ascending=False)
display(bundles.round(3))
bundles.round(4).to_csv(OUT / "rec_bundle_deals.csv", index=False)


# Bundles are built only from **1→1 rules in the actionable zone**. The compound 2→2 rules are deliberately *not* bundled:
# ~100 baskets a month is too thin to justify a permanent SKU combo, and they are just two planted bundles appearing together.
# `baskets_A_only` / `baskets_B_only` size the conversion opportunity — customers who bought one half of the bundle but not the other.

# ### 11c. App recommendations — 'frequently bought together' prompts at checkout
# Needs **high confidence** (the prompt is usually right) **and reasonable support** (it fires often enough to matter).

# In[24]:


reco = pair_rules[(pair_rules.confidence >= 0.6) & (pair_rules.support >= 0.05) & (pair_rules.lift > 1.5)].copy()
reco["fires_per_month"] = (reco["antecedent support"] * len(df_encoded)).round().astype(int)
reco["missed_add_ons"] = ((reco["antecedent support"] - reco["support"]) * len(df_encoded)).round().astype(int)
reco = reco.sort_values(["confidence", "support"], ascending=False)
reco_out = reco[["antecedent", "consequent", "confidence", "support", "lift", "fires_per_month", "missed_add_ons"]]
display(reco_out.round(4))
reco_out.round(4).to_csv(OUT / "rec_app_cross_sell.csv", index=False)


# In[25]:


# Multi-item triggers: a cart already holding two items -> recommend a third.
# Kept only when it beats the best single-item rule for the same consequent (otherwise the 1->1 rule is enough).
multi = rules[(rules.n_antecedent == 2) & (rules.n_consequent == 1) & (rules.confidence >= 0.8) & (rules.support >= 0.03)].copy()
best_single = pair_rules.groupby("consequent")["confidence"].max()
multi["best_single_conf"] = multi.consequent.map(best_single)
multi["conf_gain"] = multi.confidence - multi.best_single_conf
multi = multi[multi.conf_gain > 0.02].sort_values("confidence", ascending=False)
print(f"{len(multi)} two-item triggers add >2pp confidence over the best single-item rule")
multi[["antecedent", "consequent", "support", "confidence", "lift", "best_single_conf", "conf_gain"]].head(10).round(4)


# In[26]:


fig, ax = plt.subplots(figsize=(10, 4.8))
lab = reco_out.antecedent + "  →  " + reco_out.consequent
ax.barh(lab[::-1], reco_out.confidence[::-1], color="#4C72B0")
for k, (c_, s_) in enumerate(zip(reco_out.confidence[::-1], reco_out.support[::-1])):
    ax.text(c_ + 0.01, k, f"{c_:.0%} conf · {s_:.0%} supp", va="center", fontsize=8)
ax.set(xlim=(0, 1.15), title="'Frequently bought together' prompts (confidence ≥ 60%, support ≥ 5%)", xlabel="Confidence")
plt.tight_layout(); plt.savefig(CHARTS / "07_app_recommendations.png", dpi=150); plt.show()


# ## 12. Validation against the reference answer-key files
# `frequent_itemsets.csv` and `association_rules_full.csv` were supplied for mentor use; our independently mined results should match them exactly.

# In[27]:


import ast
parse = lambda s: frozenset(ast.literal_eval(s.replace("frozenset(", "")[:-1])) if s.startswith("frozenset") else frozenset()
ref_fi = pd.read_csv(DATA / "frequent_itemsets_reference.csv")
ref_fi["itemsets"] = ref_fi["itemsets"].apply(parse)
ref_map = dict(zip(ref_fi.itemsets, ref_fi.support.round(6)))
our_map = {k: round(v, 6) for k, v in ap_map.items()}
print("Frequent itemsets:", len(ref_fi), "in reference |", "identical:", ref_map == our_map)

ref_r = pd.read_csv(DATA / "association_rules_reference.csv")
ref_r["key"] = list(zip(ref_r.antecedents.apply(parse), ref_r.consequents.apply(parse)))
ours = rules.assign(key=list(zip(rules.antecedents, rules.consequents))).set_index("key")
m = ref_r.set_index("key")[["support", "confidence", "lift"]].join(ours[["support", "confidence", "lift"]], rsuffix="_ours", how="outer")
print("Rules:", len(ref_r), "in reference |", len(rules), "ours | in both:", m.dropna().shape[0])
max_diff = max((m[c] - m[c + "_ours"]).abs().max() for c in ["support", "confidence", "lift"])
print(f"Max absolute difference in support/confidence/lift: {max_diff:.2e}")
assert ref_map == our_map and m.dropna().shape[0] == len(ref_r) == len(rules) and max_diff < 1e-9
print("Our mining reproduces the reference answer key exactly ✔")


# ## 13. Conclusions
# 
# 1. **Reshape first.** The long transaction log (26,878 lines) became a 5,000 × 60 one-hot basket matrix; all locked sanity numbers match the spec.
# 2. **Apriori = FP-Growth on results.** Both found the same 256 frequent itemsets (60 / 78 / 86 / 30 / 2 by size) at `min_support = 0.02`. Apriori is faster at that threshold on this small catalog; FP-Growth wins clearly once support drops below ~0.5% and itemsets run into the thousands.
# 3. **980 rules with lift ≥ 1**, identical to the reference answer key.
# 4. **All six planted patterns recovered exactly** (support, confidence and lift to the spec's precision).
# 5. **Lift alone misleads.** The top-lift rules (~26.8) are compound 2→2 rules built from two co-occurring planted patterns, backed by only ~2% of baskets.
# 6. **The control pair (Banana → Shampoo) never clears 2% support** — the mined patterns are real signal.
# 7. **Actionable rules (support > 5%, lift > 2): 42**, all built on planted patterns. A bonus finding: **Cooking Oil** is a hidden third member of the Thali Staples pattern. These drive the recommendations:
#    - **Shelf placement:** six pick zones (five pairs + the Rice / Dal / Oil trio). Four of them cross category boundaries (Dairy + Bakery, Snacks + Beverages twice, Dairy + Staples), so a category-based layout would split them.
#    - **Bundles:** Ghee + Poha and Hand Sanitizer + Face Wash are the hero bundles (lift ≈ 5.1); Milk + Bread, Chips + Cola and Coffee + Cookies are everyday combos already bought together ~85–94% of the time; Rice + Dal + Oil works as a monthly-staples kit.
#    - **App prompts:** e.g. "Bought Toned Milk? Add White Bread" is right 94% of the time and fires in about a quarter of all orders.
# 
# Outputs: `outputs/*.csv` (all rules, itemsets, actionable rules, three recommendation tables, benchmark) and `charts/*.png`.
