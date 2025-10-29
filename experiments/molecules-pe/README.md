## Datasets

Download the DrugOOD datasets [here](https://drive.google.com/drive/folders/17nVALCgTz0LV8pVuoM0xQnRqwRH3Bz7a?usp=drive_link). 

## Running experiments

To run experiments on DrugOOD, cd to ./drugood and run
```
python runner.py --config_dirpath ../configs/drugood/assay --config_name SRF_gine_gin.yaml --dataset assay
python runner.py --config_dirpath ../configs/drugood/scaffold --config_name SRF_GINE_GIN.yaml --dataset scaffold
python runner.py --config_dirpath ../configs/drugood/scaffold --config_name SRF_GINE_GIN.yaml --dataset size
```

To run experiments on the Peptides-struct dataset, cd to ./peptides and run
```
python runner.py --config_dirpath ../configs/peptides --config_name SRF-BPEARL-peptides.yaml 
python runner.py --config_dirpath ../configs/peptides --config_name SRF-RPEARL-peptides.yaml 
```

## Attribution
We extend the code provided by [Kanatsoulis et al., 2025](https://github.com/ehejin/Pearl-PE) which in turn uses the code framework from the SignNet repository by [Lim et al., 2024](https://github.com/cptq/SignNet-BasisNet).