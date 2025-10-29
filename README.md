# Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks

This repository contains the source code for our manuscript [Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks](https://neurips.cc/virtual/2025/poster/116734). Our _Sketched Random Features_ enhance graph neural networks by incorporating randomized global embeddings of node features alongside traditional local representations. Such a strategy mitigates common weaknesses of GNNs such as oversquashing in the presence of informative node features.

For questions, please contact [ryien@chicago.edu](mailto:ryien@uchicago.edu).

### Getting started

To install requirements:

```bash
pip install -r requirements.txt
```

>📋  Note: Pytorch geometric (and some of its dependencies) may require a separate system specific installation process. If Pytorch Geometric cannot be installed directly with pip, please refer to the [Pytorch Geometric Installation Guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install this package.

## Quick Start
SRF integrates into existing machine learning pipelines with message-passing graph neural networks in two steps. Note that our implementation is in [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/).

First, generate SRF features during preprocessing using the `AddFeaturesTransform` class in `srf/preprocess.py`. This class can be passed as a preprocessing function to Pytorch Geometric style datasets. For example, with a [Open Graph Benchmark dataset](https://ogb.stanford.edu/):

```
from ogb.graphproppred import PygGraphPropPredDataset
transform_args = {"width": 128, "K": 256}
transform = AddFeaturesTransform(
        D_out=transform_args["width"],
        k=transform_args["K"],
        )
dataset = PygGraphPropPredDataset(
        name=dataset_name.lower(), pre_transform=transform
)
```

Second, these features should be injected into each layer of the message-passing graph neural network during training. The file `srf/main.py` contains simple examples of this injection process (as described in Section 3 of our main manuscript). Different configurations and datasets can be customized in the `config` dictionary in the main function of this file.  

## Manuscript Experiments

The following section includes instructions on reproducing the experiments in our manuscript.  

### Experiments on Synthetic Data

To run expressiveness (CSL) experiments, execute `main.py` with default settings:

```
cd srf
python main.py
```

To run [[Tree-NeighborsMatch](https://github.com/tech-srl/bottleneck)] (oversquashing) dataset, refer to its respective `README.md`:

```
cd srf/bottleneck_gnn
cat README.md
```

To run oversmoothing analysis and generate associated figures (Figure 1 in the manuscript):

```
cd srf
python oversmoothing.py
```


### Experiments on Real-World Data
Instructions to run experiments on real world graph learning tasks (Section 4.2) are contained in the `README.md` files of the appropriate subdirectories. For experiments on the Reddit datasets:

```
cd experiments/reddit-pe
cat README.md
```

For the DrugOOD and Peptides experiments:

```
cd experiments/molecules-pe
cat README.md
```

## Citation
If you find this work useful, please cite:
```
@inproceedings{
sketched-random-features,
title={Sketch-Augmented Features Improve Learning Long-Range Dependencies in Graph Neural Networks},
author={Ryien Hosseini and Filippo Simini and Venkatram Vishwanath and Rebecca Willett and Henry Hoffmann},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=gXoMU9YYdY}
}
```

## Attribution
This project builds upon several prior works, including Tree-NeighborsMatch (Alon & Yahav, 2021), Pearl (Kanatsoulis et al., 2025), SignNet (Lim et al., 2022), and SPE (Huang et al., 2024). We have modified and extended components from these repositories for our experiments.
We include use of the  [[Tree-NeighborsMatch repo](https://github.com/tech-srl/bottleneck)] by Alon and Yahav, 2021; [[Pearl repo](https://github.com/ehejin/Pearl-PE)] by Kanatsoulis et al., 2025; [[SignNet repo](https://github.com/cptq/SignNet-BasisNet)] by Lim et al., 2022; and the [[SPE repo](https://github.com/Graph-COM/SPE)] by Huang et al., 2024.
