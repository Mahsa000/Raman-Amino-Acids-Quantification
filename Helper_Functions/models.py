import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn import model_selection
from models import *
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import LeaveOneOut
from sklearn import model_selection
#define cross-validation method
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score

def select_optimal_pls_components(specss_ave, y_ave, max_components=30, n_splits=10, n_repeats=3, random_state=1):
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    mse = []

    # Calculate MSE with only the intercept
    score = -1 * cross_val_score(PLSRegression(n_components=1),
                                 specss_ave, y_ave, cv=cv, scoring='neg_mean_squared_error').mean()
    mse.append(score)

    # Calculate MSE using cross-validation, adding one component at a time
    for i in range(1, max_components + 1):
        pls = PLSRegression(n_components=i)
        score = -1 * cross_val_score(pls, specss_ave, y_ave, cv=cv,
                                     scoring='neg_mean_squared_error').mean()
        mse.append(score)

    # Plot test MSE vs. number of components
    plt.plot(mse)
    plt.xlabel('Number of PLS Components')
    plt.ylabel('PRESS')
    plt.title('Optimal PLS Components Selection')
    plt.show()

    press_values = np.array(mse)
    n_components_min_mse = np.argmin(press_values) 
    return n_components_min_mse


def cnn_AA_general(cfg, input_shape=(1024, 1), num_filters=(16, 32, 64), num_dense_filters=32, dropout=True):
    """
    General model for Raman data.
    Input: Model configurations, input_shape, num_filters, num_dense_filters, include_dropout
    Output: Model
    """
    model = models.Sequential()
    for num_filter in num_filters[:-1]:
        model.add(layers.Conv1D(num_filter, 3, activation='relu', input_shape=input_shape))
        model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(num_filters[-1], 3, activation='relu'))
    
    if dropout:
        model.add(layers.Dropout(0.2))
    
    model.add(layers.Flatten())
    model.add(layers.Dense(num_dense_filters))
    model.add(layers.Dense(6))
    
    model.summary()
    if cfg['compile']:
        model.compile(optimizer=cfg['opt'], loss='mse', metrics=['mse'])
    return model

import matplotlib.font_manager as fm

def plot_regression_results(y_true, y_pred, y_names, model_type='CNN', title=None):
    """
    Plot regression results for a given model type (CNN or PLSR).
    Input:
    - y_true: Ground-truth values (2D array)
    - y_pred: Predicted values (2D array)
    - y_names: List of output variable names
    - model_type: Type of regression model ('CNN' or 'PLSR')
    - title: Title for the plot (default: None)
    """
    r2_total = r2_score(y_true, y_pred)
    mse_total = mean_squared_error(y_true, y_pred)
    mae_total = mean_absolute_error(y_true, y_pred)
    
    print(f'r2:{r2_total:.2f}, MSE:{mse_total:.6f}, MAE:{mae_total:.6f}')
    
    for idx, y_name in enumerate(y_names):
        r2 = r2_score(y_true[:, idx], y_pred[:, idx])
        mse = mean_squared_error(y_true[:, idx], y_pred[:, idx])
        mae = mean_absolute_error(y_true[:, idx], y_pred[:, idx])
        
        print(y_name)
        print(f'r2:{r2:.2f}, MSE:{mse:.6f}, MAE:{mae:.6f}')
    
    # Load the Arial font if available
    arial_font = fm.FontProperties(fname=fm.findfont(fm.FontProperties(family="Arial")))
    
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, ax in enumerate(axs.flatten()):
        ax.scatter(y_true[:, idx], y_pred[:, idx])
        lims = [0, 0.5]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_xlabel(f'Measured {y_names[idx]} proportions', fontsize=14, fontname='Arial')
        ax.set_ylabel(f'Predicted {y_names[idx]} proportions', fontsize=14, fontname='Arial')
        
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_title(f"{y_names[idx]}\nR2={r2:.2f}, MSE={mse:.2f}, MAE:{mae:.2f}", fontsize=14, fontname='Arial')
        
        # Apply Arial font to tick labels
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(arial_font)
    
    plt.suptitle(f"{model_type} - Averaged over each group", fontsize=14, fontname='Arial')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_learning_curve(history, metric='loss', title='Model Loss', figsize=(6, 4)):
    """
    Plot the learning curve of a model.
    Input:
    - history: Keras history object
    - metric: Metric to plot (e.g., 'loss', 'mse', 'mae', etc.)
    - title: Title for the plot
    - figsize: Figure size (default: (6, 4))
    """
    df_loss_mse = pd.DataFrame(history.history)
    df_loss = df_loss_mse[['loss', f'val_{metric}']].copy()
    df_loss.rename(columns={'loss': 'train', f'val_{metric}': 'validation'}, inplace=True)
    df_loss.plot(title=title, figsize=figsize).set(xlabel='Epoch', ylabel='Loss')
    plt.show()

    
