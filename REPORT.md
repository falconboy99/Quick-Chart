# QuickCart Market Basket Analysis — Project Report

Apriori & FP-Growth on 5,000 QuickCart baskets (12 stores, November 2026, 60-SKU catalog).
Full, executed analysis: [QuickCart_Market_Basket_Analysis.ipynb](QuickCart_Market_Basket_Analysis.ipynb) (HTML copy alongside it).

## Headline results

- Every locked sanity number in the spec reproduces exactly, and the notebook `assert`s each one.
- All **6 planted patterns** are recovered with the spec's exact support, confidence and lift.
- Our mined itemsets (256) and rules (980) match the reference answer-key files to within 1e-14.
- The **highest-lift rules (≈26.8) are a trap**: compound 2→2 rules from two planted patterns in one basket, backed by only ~100 of 5,000 baskets.
- The **control pair Banana → Shampoo** appears in just 11 baskets (0.22% support, lift 0.61), far below the 2% threshold.
- **Bonus finding:** Cooking Oil 1L is a hidden third member of *Thali Staples*. Rice, Dal and Oil all pair at lift ≈ 3.7 and ~15% support.

## 1. Data and reshape

| Check | Result | Spec |
|---|---|---|
| Line items / orders / SKUs / stores | 26,878 / 5,000 / 60 / 12 | same |
| Date range | 2026-11-01 → 2026-11-30 | same |
| Basket size mean / median / min / max | 5.38 / 5 / 1 / 16 | same |
| Top-5 SKUs by support | White Bread 25.46%, Toned Milk 25.40%, Potato Chips 21.84%, Cola 21.64%, Toor Dal 20.68% | same |

There were no missing values, no duplicate (order, SKU) lines, and the names and categories agree with `dim_skus.csv`.
The long log was reshaped from line items to a basket list and then to a 5,000 × 60 one-hot matrix with `TransactionEncoder`.

## 2. Frequent itemsets: Apriori vs FP-Growth

At `min_support = 0.02` both algorithms return the identical 256 itemsets: 60 singles, 78 pairs, 86 triples, 30 of size 4 and 2 of size 5.

| min_support | itemsets | Apriori (ms) | FP-Growth (ms) |
|---|---|---|---|
| 0.02 | 256 | 24 | 77 |
| 0.005 | 2,001 | 106 | 136 |
| 0.003 | 3,922 | 431 | 230 |
| 0.002 | 7,569 | 748 | 257 |

On this small catalog at the spec threshold, mlxtend's vectorised Apriori is actually faster. FP-Growth overtakes it below about 0.5% support and is roughly 3x faster at 0.2%.
Timings are best-of-3 on the build machine and will vary, but the crossover is the point.

## 3. The 6 planted patterns

| Pattern | Rule | Support | Confidence | Lift | Baskets |
|---|---|---|---|---|---|
| Breakfast Combo | Toned Milk → White Bread | 23.84% | 93.86% | 3.69 | 1,192 |
| Thali Staples | Basmati Rice → Toor Dal | 15.34% | 77.40% | 3.74 | 767 |
| Snack-Drink Combo | Potato Chips → Cola | 19.74% | 90.38% | 4.18 | 987 |
| Ghar Breakfast | Ghee → Poha | 14.68% | 86.05% | 5.07 | 734 |
| Tea-Time | Instant Coffee → Choco Cookies | 15.20% | 85.88% | 4.84 | 760 |
| Personal Care Restock | Hand Sanitizer → Face Wash | 13.18% | 79.30% | 5.05 | 659 |

Across all 1,770 possible SKU pairs, the non-planted pairs have a median lift of 0.95. Ordinary co-purchase is basically independent, and the planted pairs stand out clearly (see `charts/05` and `charts/06`).

## 4. Why lift alone misleads

The top rule is `{Hand Sanitizer, Instant Coffee} → {Face Wash, Choco Cookies}`: lift 26.85 but support 2.06% (103 baskets).
It exists only because the Tea-Time and Personal Care patterns sometimes land in the same basket.
`Toned Milk → White Bread` has a "modest" lift of 3.69 but rests on 1,192 baskets, which makes it far more reliable and far more useful.
Rules should always be judged on support and lift together (`charts/04_lift_vs_support.png`).

## 5. Actionable rules

The spec filter (`support > 5%` and `lift > 2`) keeps **42 of 980 rules**, and every one is built on a planted pattern.
That is the 12 directed core-pair rules, 4 Rice/Dal/Oil rules, and 26 rules that add one popular item to a planted pair, such as `{Cola, Toned Milk} → Potato Chips`.
None of the compound high-lift rules survive.

## 6. Business recommendations

### Shelf placement (dark-store pick zones)
| Zone | SKUs together | Categories |
|---|---|---|
| 1 | Toned Milk + White Bread | Dairy / Bakery |
| 2 | Cola + Potato Chips | Beverages / Snacks |
| 3 | Basmati Rice + Toor Dal + Cooking Oil | Staples |
| 4 | Instant Coffee + Choco Cookies | Beverages / Snacks |
| 5 | Ghee + Poha | Dairy / Staples |
| 6 | Hand Sanitizer + Face Wash | Personal Care |

Four of the six zones cross category boundaries, so a purely category-based layout would separate items that are picked together 75–94% of the time.
Milk and bread (zone 1) should go nearest the packing station because they appear in about a quarter of all orders.

### Bundle deals
- **Hero bundles (8–10% off, home-screen promo):** Ghee + Poha and Hand Sanitizer + Face Wash. These have the strongest affinity (lift ≈ 5.1).
- **Everyday combos (3–5% off):** Milk + Bread, Chips + Cola and Coffee + Cookies. They are already bought together 85–94% of the time, so keep the discount small and use it to protect the basket.
- **Standard bundle (5–8% off):** Basmati Rice + Toor Dal. It has the most single-item buyers to convert (224 Rice-only and 267 Dal-only baskets).
- **Monthly-staples kit (~5% off):** Rice + Dal + Oil, bought together in 10.98% of baskets. It suits a subscription or "refill" offer.
- **Not bundled:** the compound 2→2 rules. About 100 baskets a month is too thin to justify a combo.

### App "frequently bought together" prompts
Prompts fire when the trigger item is in the cart. They require confidence ≥ 60%, support ≥ 5% and lift > 1.5, which gives 16 rules.

| If cart has | Suggest | Right | Fires in |
|---|---|---|---|
| Toned Milk | White Bread | 94% | 25% of orders |
| Cola | Potato Chips | 91% | 22% |
| Poha | Ghee | 86% | 17% |
| Instant Coffee | Choco Cookies | 86% | 18% |
| Face Wash | Hand Sanitizer | 84% | 16% |
| Basmati Rice | Toor Dal | 77% | 20% |

For each pair, use the direction with the higher confidence. For example, Face Wash → Sanitizer is right 84% of the time, versus 79% the other way.
In 17 cases a two-item cart lifts confidence by a further 2–4 points, for example {Cooking Oil, Instant Coffee} → Choco Cookies at 90%. These are an optional refinement.

## Deliverables

| Path | Contents |
|---|---|
| `QuickCart_Market_Basket_Analysis.ipynb` / `.html` / `.py` | Full executed analysis (notebook, HTML view, plain script) |
| `outputs/frequent_itemsets_apriori.csv` | 256 frequent itemsets |
| `outputs/association_rules_all.csv` | All 980 rules with every metric |
| `outputs/planted_patterns_verification.csv` | Answer-key comparison |
| `outputs/actionable_rules.csv` | 42 rules with support > 5% and lift > 2 |
| `outputs/rec_shelf_placement.csv`, `rec_bundle_deals.csv`, `rec_app_cross_sell.csv` | Business recommendation tables |
| `outputs/benchmark_apriori_vs_fpgrowth.csv` | Runtime benchmark |
| `charts/*.png` | 7 charts |
| `data/` | Input data (and reference answer keys) |

**How to re-run:** run `pip install mlxtend pandas matplotlib seaborn`, then run the notebook from this folder. It reads `data/` and writes `outputs/` and `charts/`.
Built with Python 3.13, pandas 3.0, mlxtend 0.25.
