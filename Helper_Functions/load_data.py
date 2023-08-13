# deal with data format differences
import numpy as np
import pandas as pd
import os
from .utils import *
from natsort import natsorted

def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f
# load any data from convrntionl Wasatch            
def data_loader_by_path(data_dir,skiprows=45): # column_name='Raw', , n_skiprows
    """
    Parameters
    ----------
    data_name: str
        Name of the set to load, transform and return

    Returns
    -------
    spec array
    wavenumber array
    id array
    """
    data_dir = data_dir
    soil_list = listdir_nohidden(data_dir)
    id_out = []
    specs = []
    gt_array = []
    for soil_id in soil_list:
        df = pd.read_csv(data_dir + '/' + soil_id, skiprows=skiprows, encoding_errors='ignore')
        columns = [c for c in df.columns if 'Raw' in c]
        spec_file = df[columns].values
        if len(specs) == 0:
            specs = spec_file
        else:
            specs = np.concatenate([specs, np.array(spec_file, dtype=float)], axis=1)

        id_out.append([soil_id[:-4]]*len(columns))#soil_id+ '_' + idx for idx in columns]
        waves_nm = df['Wavelength'].values
#         waves_cm = df['Wavenumber'].values
        waves_cm = wavelength_to_wavenumber(waves_nm, 784.816)
    return np.vstack(specs), waves_cm, waves_nm, np.array(np.hstack(id_out))
# load any ratios (y values) for amino acid mixtures
def y_loader_by_path(data_dir, index_col='vial #'):
        df = pd.read_excel(data_dir, engine='openpyxl', index_col=index_col).dropna(how='any')#openpyxl
        names = list(df. index)
        y = df.loc[names, :].values
        y_names = list(df.columns. values)
        return df, np.array(y), np.array(y_names), np.array(names) 
    
