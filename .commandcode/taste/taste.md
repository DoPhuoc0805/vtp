# Taste (Continuously Learned by [CommandCode][cmd])

[cmd]: https://commandcode.ai/

# architecture
- Separate transform and filter phases: transform functions should only create feature columns without threshold/filter logic. Confidence: 0.75
- Keep intermediate data columns (like `month`) in transform output even when removing filter logic, for potential downstream use. Confidence: 0.65
- Co-locate derived/trend features for a data source in the same module rather than creating separate modules, to share day_function and day_prefix without duplication. Confidence: 0.70

# python
- Use pd.qcut with q=n_bins, labels=False, duplicates="drop" for equal-frequency (quantile) binning of customer features. Confidence: 0.65

# code-style
- Group transform functions in `order.py` by domain with section headers: Day stage, Core (avg), Sum (total + handy), Binning (quantile). Name functions descriptively: `transform_order_<aggregation>_lxm`. Confidence: 0.65

