# A Deep Learning-based Pipeline for Whitefly Pest Abundance Estimation on Chromotropic Sticky Traps

Integrated Pest Management (IPM) is crucial in smart agriculture for sustainable pest control and crop optimization. A central aspect of IPM involves pest monitoring, often done with chromotropic sticky traps placed in insect-prone areas to assess pest populations. We introduce a deep learning-based system for counting insects in trap images, reducing the need for manual inspections and saving time and effort. Our solution also provides insect positions and confidence scores, aiding practitioners in filtering unreliable predictions.

We evaluate our approach over the PST - Pest Sticky Traps dataset, designed for whitefly counting. Results demonstrate the effectiveness of our counting strategy as an AI-based tool for pest management. 

The paper is under review to the [Ecological Informatics](https://www.sciencedirect.com/journal/ecological-informatics) journal.

## Usage
We reccomend to use the Conda virtual environment with Python 3.10 and install the required packages listed in the `requirements.txt` file with pip.
```
1. conda create -n insect-counting python=3.10
2. conda activate insect-counting
3. pip install -r requirements.txt
```

Then, prepare the data using the `reproduce.sh` script.
```
1. cd data
2. ./prepare_data.sh
```

To reproduce the experiments of the paper, run the following commands:
```
1. ./reproduce.sh
2. ./reproduce_eval.sh
```

## Dataset
[PST - Pest Sticky Traps](https://zenodo.org/record/7801239)

## Contacts
Luca Ciampi - luca.ciampi@isti.cnr.it