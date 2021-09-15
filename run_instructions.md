## Setup

To access data, run: ```ssh -i <path_to_pem_key> -L 9200:localhost:9200 uchicago@188.166.218.114```
Place uchicago.ini file in base directory. 

## Usage

### scrape.py

Functions for acquiring article text (and credibility labels, if available)

### features.py

Functions for extracting features from input articles. Features extracted include sentiment features, similarity features, entity, and quotation features.

### classifier.py

Functions for training a classifier to predict article credibility based on article features. Functions can either call functions in `features.py` to generate feature table, or use existing csv in `data/` directory.  

### pipeline.py

Script that can be easily run to predict credibility of new articles using feature functions in `features.py` and a classifier from `classifier.py`

To use:
``` python pipeline.py --dataset <dataset name, defaul emm_data> --n <number of articles>```

## Directories

### notebooks

Directory containing scratch jupyter notebooks

### data

This directory holds data files (as csv) storing a generated feature table. The remote repository only stores the most recent feature table, `features.csv`.