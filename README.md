# Correlation and Relationships in Data — student notebooks

Coffee-shop practice materials: four Jupyter notebooks that walk from scatter plots to correlation,
causation and machine learning, all using the same synthetic café dataset.

## Files

```
generate_cafe_data.py            # regenerates all CSVs (documented, change SEED to redraw)
cafe_daily.csv                   # main dataset — 731 days of café operations
cafe_promo_experiment.csv        # randomised A/B promo test (observational vs experimental)
cafe_two_stores.csv              # two locations (Simpson's paradox)
data_dictionary.md               # every column + which notebook uses it

1_relationships.ipynb            # scatter plots, independent/dependent variables
2_covariance_correlation.ipynb   # covariance, Pearson's r, limitations
3_causation.ipynb                # confounding, observational vs experimental, spurious correlation
4_regression_ml.ipynb            # regression + feature selection
```

Work through the notebooks in order. Each one explains the idea, gives the formulas, runs the code,
and then explains how to read the result.

## Running

Keep the CSVs in the same folder as the notebooks. Run the **Setup** cell at the top of each notebook
first (and again after restarting the kernel). The notebooks ship without saved outputs, so run each
cell yourself and compare the result with the notes.

Requirements: `pandas numpy matplotlib seaborn scipy scikit-learn`

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
```

## Note

All data is synthetic and every relationship is engineered on purpose; the generator contains no real people or events.
