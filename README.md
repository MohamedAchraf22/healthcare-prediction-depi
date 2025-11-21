![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-%23150458.svg?logo=pandas&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)


This project implements a healthcare predictive modeling pipeline developed for the Digital Egypt Pioneers Initiative (DEPI). The primary objective is to preprocess medical data, explore feature relationships, and build classification models capable of identifying patients at elevated health risk based on clinical, and lifestyle attributes.
## How to use

### Clone repo
using HTTPS 
```
git clone https://github.com/forge34/healthcare-prediction-depi.git
```

using SSH
```
git clone git@github.com:forge34/healthcare-prediction-depi.git
```

### Install packages
```
pip install ./requirements
```
or 

```
conda env create -f environment.yml
```

### Run the pipeline
to run preprocessing  

```
python ./src/preprocessing.py
```

to generate visualizations
```
python ./src/visualization.py
```
