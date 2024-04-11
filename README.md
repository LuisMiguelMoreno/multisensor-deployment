![](figure.png)

# multisensor-deployment

This repository contains the code for the paper "*Evolutionary optimization of spatially-distributed multi-sensors placement for indoor surveillance environments with security levels*". In this paper, our goal is to optimize the placement of sensors (smoke detectors, cameras, seismic detectors, etc) in a surveillance environment; to achieve this, we employ evolutionary meta-heuristics.

## How to run

To reproduce the paper's results, simply:

```bash
git clone https://github.com/LuisMiguelMoreno/multisensor-deployment
cd multisensor-deployment
pip install -r requirements.txt
```

Each experiment can be reproduced by running its corresponding script. The scripts are contained in the `experiments` folder and can be run as follows:

```bash
python experiments/Experiment_2_EA.py
python experiments/Experiment_2_GRASP.py
```

## Citing the paper

To cite the paper, please use the following BibTeX entry:

```bibtex
foobar
```
