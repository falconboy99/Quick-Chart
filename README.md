# QuickCart Market Basket Analysis

Association-rule mining (Apriori & FP-Growth) on 5,000 QuickCart baskets across 12 stores in November 2026, using the same 60-SKU catalog as the earlier QuickCart projects.
The goal is to find which products are bought together and turn the strongest rules into shelf-placement, bundle-deal and "frequently bought together" recommendations.

- **Findings & recommendations:** [REPORT.md](REPORT.md)
- **Full analysis:** [QuickCart_Market_Basket_Analysis.ipynb](QuickCart_Market_Basket_Analysis.ipynb), also available as `.html` and `.py`
- **Project spec:** [docs/QuickCart_MarketBasket_Project_Spec.pdf](docs/QuickCart_MarketBasket_Project_Spec.pdf)

## Highlights
- Every sanity number in the spec reproduces exactly: 26,878 line items, 5,000 baskets, 256 frequent itemsets and 980 rules.
- All 6 planted co-purchase patterns are recovered with their exact support, confidence and lift.
- The highest-lift rules (lift ≈ 26.8) rest on only ~2% of baskets. This trap is explained in the notebook.
- A control pair, Banana → Shampoo, never reaches the 2% support threshold, which shows the patterns are real.
- Bonus finding: Cooking Oil is a hidden third member of the "Thali Staples" pattern.

## Layout
```
data/      input transactions, SKU catalog, reference answer keys
outputs/   mined itemsets, all rules, actionable rules, recommendation tables, benchmark
charts/    7 PNG charts
```

## Run it
```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace QuickCart_Market_Basket_Analysis.ipynb
```
The notebook reads `data/` and regenerates `outputs/` and `charts/`.
