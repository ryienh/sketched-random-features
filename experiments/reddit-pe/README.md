## Running Experiments

To run SRF, on the REDDIT-BINARY and REDDIT-MULTI datasets, run
```
python -m train.reddit --config reddit-m.yaml 
python -m train.reddit --config reddit-b.yaml
```

SRF variants and baselines can be run by editing the configuration (`.yaml`) files accordingly.

## Attribution
We extend the code provided by [Kanatsoulis et al., 2025](https://github.com/ehejin/Pearl-PE) which in turn uses the code framework from the SignNet repository by [Lim et al., 2024](https://github.com/cptq/SignNet-BasisNet).
