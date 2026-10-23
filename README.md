# KB-DSS

Repository: https://github.com/Vineethsamba/kb-dss

Reproduction code for:

**V. K. Samba**, “A Knowledge-Based AI Decision-Support Model for Predicting Digital Transformation Readiness From Enterprise Innovation Data.”

KB-DSS injects five expert pillars, six signed fuzzy IF–THEN rules, and an expert index into a gradient-boosting learner, and measures the value of that knowledge with a fidelity protocol.

## Run

```bash
python -m pip install -r requirements.txt
python dtr.py
```

This generates the synthetic benchmark of 2,000 firms (seed 42), runs every experiment, and writes:

- `outputs/fig1.png` — framework
- `outputs/fig2.png` — learning curves, fidelity map, permutation importance
- `outputs/metrics.json` — Table 1, Table 2, and sweep metrics

Typical runtime is a few minutes on a laptop CPU. A self-test of the knowledge-layer invariants runs before every experiment.

The IEEE manuscript lives in `paper.tex` / `paper.pdf`. Tables in the manuscript are from the original study run; `python dtr.py` regenerates the same protocol with seed 42.

## Real data

```bash
python dtr.py --data firms.csv
```

Required columns (indicators in `[0, 1]`):

`data_maturity`, `analytics_capability`, `ai_adoption`, `cloud_adoption`, `data_governance`, `leadership_commitment`, `digital_skills`, `rd_intensity`, `innovation_output`, `entrepreneurial_orientation`, `legacy_debt`, `regulatory_intensity`, `log_size`, `sector`, `y`

- `sector` is an integer in `{0,1,2,3,4}`: manufacturing, services, finance, healthcare, other
- `y` is `Low` / `Medium` / `High` or `0` / `1` / `2`
- `log_size` is standardized log firm size

## License

MIT. See `LICENSE`.
