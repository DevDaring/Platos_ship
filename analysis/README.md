# analysis/ — regenerate every number, table and figure in the paper

One deterministic path from the released per-trial logs to the manuscript's
reported values. No API access or GPU is required; these read the parquet files
in `results/` and `Code_Phase_2/results/`.

```bash
python3 analysis/verify_paper_numbers.py   # recompute every headline number -> paper_numbers.json
python3 analysis/make_tables.py            # emit the paper's LaTeX data tables
python3 analysis/make_figures.py           # regenerate the figures
```

`verify_paper_numbers.py` is the single source of statistical truth. It:

- separates the perturbed-GSM8K pool from the main 300-question pool (they share
  condition IDs but are different questions);
- excludes the superseded 50-question cross-validation focal;
- reports trial-level and question-level pairings separately;
- declares each multiplicity family and applies Holm within it;
- computes the capability correlation on unrounded rates, with an exact
  permutation p-value, leave-one-model-out and drop-the-weakest sensitivity;
- tests answer-level consensus within C4;
- bounds the effect of unrecovered parse failures.

The manuscript `\input`s the generated tables, so it cannot drift from the logs.
Requires: pandas, numpy, scipy, pyarrow, matplotlib.
