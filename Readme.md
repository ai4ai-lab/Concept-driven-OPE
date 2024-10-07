This repository consists of the code for our paper: "Concept-driven Off Policy Evaluation". The repository is split into 4 sections: The 2 environments: Synthetic WindyGridworld and MIMIC-III dataset. The repository "Result" consists of the quantitative outputs of our experiments, while "Plots" consist of the code to replicate the plots of our experiments.

## Dataset generations:
To generate the dataset for WindyGridworld: Run the code /WindyGridworld/trajectories_generation.py

MIMIC: The dataset is publicly available, the excel files can be downloaded from: https://www.nature.com/articles/s41591-018-0213-5#data-availability. Post downloading the dataset: Run /MIMIC/preprocess.py to generate the trajectories.

## Known Concept Experiments:
The concepts are predefined, with further details in the following files. 

WindyGridworld: /WindyGridworld/knownconceptswindygridworld.py  

MIMIC: /MIMIC/knownmimic.py 

## Unknown Concept Experiments:
The files are divided into CBM training, and post which performing interventions based on learnt concepts. Relevant files:

WindyGridworld: /WindyGridworld/unknownconceptswindygridworld.py # The first 

MIMIC: /MIMIC/unknownmimic.py

## Intervention Experiments:
WindyGridworld: Qualitative: Expert concepts are predefined and mentioned. Parameterized concepts are labeled based on the magnitude of the weights. Quantitative metrics follow in the file: /WindyGridworld/interventionswindygridworld.py

MIMIC: Qualitative: Urine output based segregation of trajectories. Quantitative: Interventions based on the definitions in the paper. Relevant file: /MIMIC/interventions.py


