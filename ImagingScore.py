from __future__ import division, print_function, unicode_literals

import numpy as np
import pandas as pd

# IMPORT DATABASE (see database example for required structure)
data = pd.read_excel(r'data/database_structure.xlsx') 

# STRATIFY OS (<> 12 weeks)
data['OS_bin'] = np.where(data['OS_weeks'] > 12 ,1,0) 

# REMOVE ENTRIES WITH MISSING OS AND/OR MORE THAN 2 MISSING FEATURES, IMPUTE REMAINING MISSING VALUES
data = data.dropna(subset=['OS_bin'])
data = data.dropna(thresh=10) # keeps only entries with at least 7 non-Na values
#var_impute = ['ECOG_PS', 'BMI', 'SD_max', 'WB_MATV', 'VISC_density_HU', 'Msites', 'days_since_diagnosis']
#data[var_impute] = data[var_impute].fillna((data[var_impute].mean()), inplace=True)

# STANDARDIZE VALUES WITH PRE-FITTED POWERTRANSFORMER
from sklearn.preprocessing import PowerTransformer
import joblib
power_transformer = joblib.load('power_transformer.joblib')
var_standard = ['OS_weeks', 'ECOG_PS', 'BMI', 'SD_max', 'WB_MATV', 'VISC_density_HU', 'Msites', 'days_since_diagnosis']
data[var_standard] = power_transformer.transform(data[var_standard])

# LOAD IMAGING SCORE AND APPLY
Imaging_Score = joblib.load('ImagingScore.joblib')

# EVALUATE PREDICTIONS USING GROUND-TRUTH THROUGH CONCORDANCE INDEX AND 100 BOOTSTRAPS
from scipy import stats
from lifelines.utils import concordance_index
from sklearn.utils import resample

ci_results = np.array([])
bootstraps = 100

for i in range(bootstraps):
    rs = data.sample(frac = 1, replace=True)
    groundtruth = rs['OS_weeks'].copy()
    probability_predictions = Imaging_Score.predict_proba(rs.drop(['patient_ID','OS_weeks', 'OS_bin'], axis=1))
    Cindex=concordance_index(groundtruth.to_numpy(), probability_predictions[:,1])
    ci_array = np.array([Cindex])     
    ci_results = np.append(ci_results, ci_array)


confidence_level = 0.95
confidence_intervals = np.apply_along_axis(lambda col: stats.t.interval(confidence_level, len(col)-1, loc=np.mean(col), scale=stats.sem(col)), axis=0, arr=ci_results)

# PRINT RESULTS
print("Mean concordance index:")
print(np.mean(ci_results))
print("Concordance index standard deviation:")
print(np.std(ci_results))
print("Concordance index 95% Confidence Intervals:")
print(confidence_intervals)
