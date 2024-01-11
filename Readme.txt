DISCLAIMER: THE CODE IN THIS REPOSITORY SHOULD NOT BE USED FOR ANY CLINICAL IMPLEMENTATIONS. 
The trained ImagingScore available in this repository is only provided for further validation of its performance on new datasets, it is not intended for clinical use. 

Requirements:
Python version 3.12
	numpy
	pandas
	sklearn
	joblib
	scipy
	lifelines

To validate the ImagingScore on your dataset, run 'ImagingScore.py' on a dataset that follows the 'database_structure.xlsx' format. 
OUTPUT: mean concordance index with standard deviation and 95% confidence interval on 100 (default) bootstraps.
