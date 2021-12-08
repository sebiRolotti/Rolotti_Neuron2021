# Analysis code for Rolotti et al. *Neuron* 2021

This repository contains code used to ultimately generate all figures from:

Rolotti S.V., Ahmed, M.S., Szoboszlay M., Geiller T., Negrean A., Blockus H., Gonzalez K.C.,
Sparks F.T., Solis Canales A.S., Tuttman A.L., Peterka D.S., Zemelman B.V., Polleux F., & Losonczy A.
Local feedback inhibition tightly controls rapid formation of hippocampal place fields.
*Neuron* 2021.

## Overview
The code contained here includes: 

* preprocessing steps after
[Suite2p](https://github.com/MouseLand/suite2p) motion correction and signal extraction.
This includes spike deconvolution and place field calculations.

* A minimal version of the internal Losonczy Lab repo with all code necessary to enable pre-processing and analysis steps.
Note that this does not include the SQL database needed for storing all experiment metadata, these details would need to be filled in
for your own database at lab_repo.classes.database.ExperimentDatabase.connect().
See an older but far more thorough version of this repo [here](https://github.com/jzaremba/Zaremba_NatNeurosci_2017) for more details about installation and use.

* A dedicated repo for the classes and analyses specific to the various forms of induction experiments included in the paper.

* Notebooks for calculating and plotting metrics to actually generate the panels of each figure (supplemented/accelerated by pre-compute in group_scripts)

Repository layout:

    .
    ├── pre_processing/               # Files for spike deconvolution and place field detection
    ├── lab_repo/                     # Core mwe code for all lab-general processing and analysis
    ├── induction_analysis/           # Classes, pre-processing, analysis, and plotting specific to induction experiments
    ├── notebooks/                    # Jupyter notebooks for generating figures
    └── README.md

### Raw data

Note that the raw data and metadata are not included here.
This means that many scripts will not run immediately, as they currently include calls to data/metadata that are not included. In particular, metadata calls require a SQL database instance, while data calls would require the ROIs and extracted signals for each experiment as well as the corresponding behavior data.

These steps are retained here to demonstrate what data was included in each analysis in the paper as well as to serve as a template for
your own data/analysis. To request the data from these experiments, or for more information on how to run your own analysis with these tools, please contact the authors.
