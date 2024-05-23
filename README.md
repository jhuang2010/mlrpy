# mlrpy

A gui programme to apply Machine Learning (ML) to data and produce predictive regression models.

## Description

The app loads data file in .csv and extract the data using the header. The data are split into training and 
testing sets. With the training set, models are trained to take inputs parameters and predict output parameters. 
Trained models are 
deployed as .gz files. With the testing set, predictions of output parameter are made from the input parameters 
using the trained models. 
Predicted results are plotted against the real data with error estimated. Details of the project can be 
found in 
ML_Regressor.md. 

## Getting Started

### Dependencies

* python3.12, scikit-learn, pandas, matplotlib
* Windows, Linux

### Installing

* Download the project zip file and unzip locally
* Install the dependencies from the requirement file
```
pip3 install -r /path/to/requirements.txt
```

### Executing program

* Run the following script

```
python3 /path/to/MLR_GUI.py
```

## Future Work
1.	Add more data file types, e.g. .txt and .xlsx and consider categorical data
2.	Read parameters and units from headers and automate plot range for data
3.	Add ANN and CNN to the ML models
4.	Separate API and GUI code
5.	Add web service 

## License

This project is licensed under the MIT License - see the LICENSE.md file for details
