import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pywt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.impute import KNNImputer
from matplotlib.ticker import MaxNLocator
from scipy.signal import savgol_filter
from matplotlib import cm, colormaps

def plot_specs(specs, w=None, id_list=None, title=None, data=None):
    """
    Plot spectral data.

    Parameters:
        specs (array-like): Array of spectra data.
        w (array-like, optional): Array of wavenumbers. Default is None.
        id_list (array-like, optional): List of IDs. Default is None.
        title (str, optional): Plot title. Default is None.
        data (str, optional): Type of data (Raman or IR). Default is None.
    """
    if w is None:
        w = range(len(specs[0]))
    if id_list is None:
        id_list = range(len(specs))
        
    fig = plt.figure()
    grid = plt.GridSpec(1, 1, wspace=0.08, hspace=0.15)
    ax = fig.add_subplot(grid[0, 0])
    for sample_num in range(len(specs)):
        plt.plot(w, specs[sample_num], label=id_list[sample_num], linewidth=0.9)
    if data == 'Raman':
        plt.xlabel('Raman shift ($cm^{-1}$)', fontsize=14)
        plt.ylabel('Raman intensity ($a.u.$)', fontsize=14)
        plt.title(title, fontsize=16)
    if data == 'IR':
        plt.xlabel('Wavenumber ($cm^{-1}$)')
        plt.ylabel('Absorbance')
        plt.title(title)
        fig.subplots_adjust(right=0.95, left=0.20, bottom=0.15, top=0.9)
        fig.set_size_inches(9, 6)
        

def plot_mineral_peak_elimination(specss_m, w_cm):
    pivot1 = np.arange(389, 427)
    pivot2 = np.arange(79, 133)
    pivot3 = np.arange(0, 79)
    pivot4 = np.arange(197, 223)
    
    pivot5 = np.arange(814, 836)
    pivot6 = np.arange(604, 624)
    pivot7 = np.arange(443, 460)
    pivot8 = np.arange(241, 268)
    
    mineral_indices = (pivot1, pivot2, pivot3, pivot4, pivot5, pivot6, pivot7, pivot8)
    rem_indices = np.concatenate(mineral_indices)
    
    specss_m_new = np.full(specss_m.shape, np.nan)
    for i in range(len(specss_m)):
        if i not in rem_indices:
            specss_m_new[i] = specss_m[i]
    
    first_non_nan_index = np.argmax(~np.isnan(specss_m_new))
    
    plt.figure(figsize=(15, 5))
    tick_label_size = 11.43
    name_size = 18
    
    fig, ax1 = plt.subplots()
    
    ax1.plot(w_cm, specss_m, color='black', label="Baseline")
    ax1.plot(w_cm[pivot1], specss_m[pivot1], color='red', label='Mineral Peaks')
    ax1.plot(w_cm[pivot2], specss_m[pivot2], color='red')
    ax1.plot(w_cm[pivot3], specss_m[pivot3], color='red')
    ax1.plot(w_cm[pivot4], specss_m[pivot4], color='red')
    ax1.plot(w_cm[pivot5], specss_m[pivot5], color='red')
    ax1.plot(w_cm[pivot6], specss_m[pivot6], color='red')
    ax1.plot(w_cm[pivot7], specss_m[pivot7], color='red')
    ax1.plot(w_cm[pivot8], specss_m[pivot8], color='red')
    
    ax1.set_xlabel('Wavenumbers (cm$^\mathdefault{-1}$)', fontsize=name_size, font='Arial', labelpad=2)
    ax1.set_ylabel('Normalized Raman intensities (a.u.)', fontsize=name_size, font='Arial', labelpad=2)
    ax1.legend(loc='upper left')
    
    ax2 = ax1.twinx()
    
    ax2.plot(w_cm[first_non_nan_index:], specss_m_new[first_non_nan_index:], color='gray', label='Baseline (Scaled)', linestyle='dashed')
    plt.tick_params(axis='both', which='both', labelsize=tick_label_size)
    ax1.legend().set_visible(False)
    
    plt.title("Mineral peak elimination and residual representation")

def mineral_peaks_removal_impute(specs_ex, n_neighbors=2, weights="uniform"):
    pivot1 = np.arange(389, 427)
    pivot2 = np.arange(79, 133)
    pivot3 = np.arange(0, 79)
    pivot4 = np.arange(197, 223)
    
    pivot5 = np.arange(814, 836)
    pivot6 = np.arange(604, 624)
    pivot7 = np.arange(443, 460)
    pivot8 = np.arange(241, 268)
    
    mineral_indices = (pivot1, pivot2, pivot3, pivot4, pivot5, pivot6, pivot7, pivot8)
    rem_indices = np.concatenate(mineral_indices)
    
    specs_rem = np.full(specs_ex.shape, np.nan)
    for spectrum in range(specs_ex.shape[0]):
        for i in range(specs_ex.shape[1]):
            if i not in rem_indices:
                specs_rem[spectrum, i] = specs_ex[spectrum, i]

    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    specs_ex = imputer.fit_transform(specs_rem)
    return specs_ex

def wavenumber_to_wavelength(wavenumbers, excitation_wavelength):
    """Convert wavenumber to wavelength basis."""
    return 1 / ((1 / excitation_wavelength) - wavenumbers * 1E-7)

def wavelength_to_wavenumber(wavelengths, excitation_wavelength):
    """Convert wavelength to wavenumber basis."""
    return ((1 / excitation_wavelength) - (1 / wavelengths)) * 1E7

def dwt_multilevel_filter(spectrum, wavelet, scale, apprx_rm, low_cut, high_cut):
    """
    Apply DWT multilevel filter to spectrum.

    Parameters:
        spectrum (array-like): Input spectrum.
        wavelet (str): Wavelet name.
        scale (int): Number of DWT levels.
        apprx_rm (bool): Whether to remove approximation coefficients.
        low_cut (int): Low cut-off level.
        high_cut (int): High cut-off level.

    Returns:
        array-like: Filtered spectrum.
    """
    coeffs = pywt.wavedec(spectrum, wavelet, level=scale)
    for c_ix in range(len(coeffs)):
        if (c_ix == 0 and apprx_rm) or (c_ix > 0 and c_ix <= low_cut) or c_ix > scale - high_cut:
            coeffs[c_ix] = np.zeros_like(coeffs[c_ix])
    return pywt.waverec(coeffs, wavelet)

def dwt_iterative_bg_rm(spectrum, wavelet, scale, iterations):
    """
    Iterative DWT background removal.

    Parameters:
        spectrum (array-like): Input spectrum.
        wavelet (str): Wavelet name.
        scale (int): Number of DWT levels.
        iterations (int): Number of iterations.

    Returns:
        tuple: Spectrum with background removed and background approximation.
    """
    bg_approx = spectrum    
    for ix in range(iterations):
        apprx_ix = dwt_multilevel_filter(bg_approx, wavelet, scale, apprx_rm=False, low_cut=0, high_cut=scale)
        if bg_approx.shape[0] % 2 == 0:
            bg_approx = np.minimum(bg_approx, apprx_ix)
        else:
            bg_approx = np.minimum(bg_approx, apprx_ix[:-1])
    spectrum_bg_removed = spectrum - bg_approx
    return spectrum_bg_removed, bg_approx

def remove_baseline_dwt(spec, reps=2, level=8, wavelet='sym5'):
    """
    Remove baseline using DWT method.

    Parameters:
        spec (array-like): Input spectrum.
        reps (int, optional): Number of repetitions. Default is 2.
        level (int, optional): DWT level. Default is 8.
        wavelet (str, optional): Wavelet name. Default is 'sym5'.

    Returns:
        array-like: Spectrum with baseline removed.
    """
    spec_clean = spec.copy()
    for _ in range(reps):
        decomposed = pywt.wavedec(spec_clean, wavelet, level=level)
        for idx in range(1, level + 1):
            decomposed[idx] = None
        reconstructed = pywt.waverec(decomposed, wavelet)
        spec_clean = np.minimum(spec_clean, reconstructed[:len(spec)])
    return spec - spec_clean[:len(spec)]

def plot_histogram(df_gt, n_rows, n_columns, title=None):
    """
    Plot histograms of DataFrame columns.

    Parameters:
        df_gt (DataFrame): Input DataFrame.
        n_rows (int): Number of subplot rows.
        n_columns (int): Number of subplot columns.
        title (str, optional): Plot title. Default is None.
    """
    fig, ax = plt.subplots(n_rows, n_columns, figsize=(15, 10), sharex='col', sharey='row')
    axis_label_size = 18
    label_padding = 2.54
    font = 'Arial'

    m = 0
    for i in range(n_rows):
        for j in range(n_columns):
            df_gt.hist(column=df_gt.columns[m], bins=12, ax=ax[i, j], alpha=0.5, color='mediumslateblue')
            ax[i, j].set_title('') 

            ax[i, j].text(0.02, 0.98, df_gt.columns[m], verticalalignment='center', horizontalalignment='left',
                          transform=ax[i, j].transAxes, fontsize=axis_label_size, weight='book', color='black')

            m += 1

            if j == 0:
                ax[i, j].set_ylabel('Frequency x $10^2$', fontsize=axis_label_size, fontname=font, labelpad=label_padding)

            if i == n_rows - 1:
                ax[i, j].set_xlabel('Measured proportions (wt%)', fontsize=axis_label_size, fontname=font, labelpad=label_padding)
            ax[i, j].yaxis.set_major_locator(ticker.FixedLocator(ax[i, j].get_yticks()))
            ax[i, j].set_yticklabels([str(int(tick/100)) for tick in ax[i, j].get_yticks()])
            ax[i, j].grid(False)

    if title is not None:
        fig.suptitle(title, fontsize=axis_label_size, fontname=font)

    for ax_row in ax:
        for ax_single in ax_row:
            ax_single.tick_params(axis='both', which='major', labelsize=axis_label_size)

    plt.tight_layout()

    for ax_row in ax:
        for ax_single in ax_row:
            ax_single.spines['right'].set_visible(False)
            ax_single.spines['top'].set_visible(False)
    plt.show()

def plot_wavelet_coeffs(specss, w_cm, scale_values, wavelet='sym5'):
    font = 'Arial'
    axis_label_size = 16
    tick_label_size = 11.43

    x = w_cm

    num_plots = len(scale_values)
    fig, axs = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    for i, scale in enumerate(scale_values):
        row = i // 3  # Calculate the row index
        col = i % 3   # Calculate the column index
        spectrum_appx_coeffs = pywt.wavedec(specss[5, :], wavelet, level=scale)[0]
        axs[row, col].plot(spectrum_appx_coeffs, color='#238A8DFF')
        n_features2 = spectrum_appx_coeffs.shape[0]
        axs[row, col].set_title(f'Scale {scale}: {n_features2} Features', fontsize=axis_label_size, fontname=font)
        axs[row, col].set_xlabel('CCD pixel count', fontsize=axis_label_size, fontname=font)
        axs[row, col].set_ylabel('Normalized Raman intensities (a.u.)', fontsize=axis_label_size, fontname=font)
        axs[row, col].tick_params(axis='both', labelsize=tick_label_size)
        axs[row, col].yaxis.set_major_locator(MaxNLocator(integer=True))

    for i, ax in enumerate(axs.flat):
        row = i // axs.shape[1]  # Calculate the row index
        col = i % axs.shape[1]   # Calculate the column index

        if col != 0:  # check if it's not the first column
            ax.set_ylabel('')  # remove the y-axis label for inner plots
        else:
            ax.set_ylabel('Raman intensities (a.u.)', fontsize=axis_label_size, fontname=font)

        if row != axs.shape[0] - 1:  # check if it's not the last row
            ax.set_xlabel('')  # remove the x-axis label for inner plots

    for ax in axs.flat:
        ax.spines['right'].set_visible(False)  # Remove right border
        ax.spines['top'].set_visible(False)    # Remove top border
        ax.tick_params(axis='both', labelsize=tick_label_size)

    plt.tight_layout()
    plt.show()

def plot_spectrum_correlations_subplots(specs_aa, y_aa, w_cm, y_names_aa, specs_b, y_b, specs_mr, y_mr):
    plt.rcParams['font.family'] = 'Arial'

    tick_label_size = 11.43

    def plot_spectrum_correlations(ax, x, y, wavelengths, color=None):
        correlations = np.array([np.corrcoef(wl, y)[0][1] for wl in x.T])
        ax.plot(wavelengths, correlations, color=color)
        ax.plot(wavelengths, np.zeros_like(correlations), ls='--', c='gray', alpha=0.4)
        ax.set_ylim([-1, 1])
        ax.set_xlabel('Wavenumber (cm$^\mathdefault{-1}$)', fontsize=18)
        ax.set_ylabel('Correlation Coefficient', fontsize=18)
        ax.tick_params(axis='both', which='both', labelsize=tick_label_size)

    fig, axes = plt.subplots(6, 3, figsize=(18, 18))

    for idx in range(6):
        plot_spectrum_correlations(axes[idx, 0], specs_aa, y_aa[:, idx], w_cm, color='red')
        axes[idx, 0].set_title(f"{y_names_aa[idx]} in AA", fontsize=18, pad=0)

    for idx in range(6):
        plot_spectrum_correlations(axes[idx, 1], specs_b, y_b[:, idx], w_cm, color='green')
        axes[idx, 1].set_title(f"{y_names_aa[idx]} in AAF", fontsize=18, pad=0)

    for idx in range(6):
        plot_spectrum_correlations(axes[idx, 2], specs_mr, y_mr[:, idx], w_cm, color='blue')
        axes[idx, 2].set_title(f"{y_names_aa[idx]} in AAM", fontsize=18, pad=0)

    for ax in axes.flat:
        ax.label_outer()
        ax.tick_params(axis='both', labelsize=tick_label_size)

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    plt.tight_layout()
    plt.show()

def correlation(df, title=None):
    """
    Plot correlation heatmap of DataFrame columns.

    Parameters:
        df (DataFrame): Input DataFrame.
        title (str, optional): Plot title. Default is None.
    """
    corr = pd.DataFrame(df).corr()
    fig = plt.figure(figsize=(8, 8))
    r = sns.heatmap(corr, cmap="OrRd_r", alpha=0.9)
    r.set_title(title)

def plot_spectrum_correlations(x, y, wavelengths, color=None):
    """
    Plot correlations between spectra and a target variable.

    Parameters:
        x (array-like): Spectral data.
        y (array-like): Target variable data.
        wavelengths (array-like): Wavelength values.
        color (str, optional): Plot color. Default is None.
    """
    correlations = np.array([np.corrcoef(wl, y)[0][1] for wl in x.T])
    plt.figure(figsize=(18, 5))
    plt.plot(wavelengths, correlations, color=color)
    plt.plot(wavelengths, np.zeros_like(correlations), ls='--', c='k')
    plt.ylim([-1, 1])
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Correlation Coefficient')

def norm_mean(x): 
    """Normalize data using mean and standard deviation."""
    return (x - np.mean(x)) / np.std(x)

def norm_min(x): 
    """Normalize data to the range [0, 1]."""
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def generate_noisy_spec(x_train, y, names, replicas, noise_generator=None, noise_amp =1, pivot1=805, pivot2=840):
    """Generate noisy spectra and repeat each one, replicas times with a random generator.
    """
    aug_specs = []
    aug_y = np.zeros((y.shape[0]*replicas, y.shape[1]))
    aug_names = np.zeros((names.shape[0]*replicas))
    new_specs = np.zeros((x_train.shape))
    new_noise = np.zeros((x_train.shape))
    for idx in range(replicas):
        for spectrum in range(x_train.shape[0]):
            SNR = np.divide(np.max(x_train[spectrum]), np.std(x_train[spectrum,pivot1:pivot2]))
            new_noise[spectrum, :] = noise_amp*(np.ptp(x_train[spectrum])/SNR)*(noise_generator(x_train.shape[1]) - 0.5) 
            new_specs[spectrum, :] = x_train[spectrum, :] + new_noise[spectrum, :]
        aug_specs.append(new_specs)
        aug_y = np.concatenate([y] * replicas, axis=0)
        aug_names = np.concatenate([names] * replicas, axis=0)
    return np.vstack(aug_specs), aug_y, aug_names  

def plot_augmented_specs(specs_aa, aug_specs, w_cm):
    ncolors = 10**3
    color_cycle = 5
    color_list = plt.cm.viridis(np.linspace(0, 0.9, color_cycle))[::-1]
    plt.rcParams['font.family'] = 'Arial'

    font = 'Arial'
    axis_label_size = 18
    tick_label_size = 11.43

    fig, axs = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    n = 0
    for i in range(aug_specs.shape[0]):
        x = np.linspace(min(w_cm), max(w_cm), aug_specs.shape[1])
        axs[0].plot(x, specs_aa[i] / 10000, color=color_list[n])
        axs[1].plot(x, aug_specs[i] / 10000, color=color_list[n])
        n += 1
        if n == color_cycle:
            n = 0

    axs[0].set_title('(a) Amino acid (AA) mixtures', fontsize=axis_label_size)
    axs[1].set_title('(b) AA mixtures with Noise', fontsize=axis_label_size)

    for ax in axs.flat:
        ax.set_xlabel('Wavenumbers (cm$^{-1}$)', fontsize=axis_label_size)
        ax.set_ylabel('Raman intensities (a.u.) x 10$^\mathdefault{4}$', fontsize=axis_label_size)
        ax.tick_params(axis='both', labelsize=tick_label_size)

    for ax in axs.flat:
        ax.label_outer()
    for ax in axs.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)

    plt.show()



def normalize_array(data):
    """
    Normalize an array of data between 0 and 1.

    Parameters:
    - data (array-like): The input data to be normalized.

    Returns:
    - normalized_data (array-like): The normalized data.
    """
    data_min = np.min(data)
    data_max = np.max(data)
    
    if data_max == data_min:
        return np.zeros_like(data)
    else:
        return (data - data_min) / (data_max - data_min)

def spectral_ratio_color_coding(ax, wavelengths, x, y, axis_title, sample_ids=None, is_first_col=False):
    """
    Plot spectra and color code by the associated measured proportions.

    This function plots multiple spectra on a single plot, with each spectrum
    being color-coded based on its associated measured proportions value.

    Parameters:
    - wavelengths (array-like): The wavelengths corresponding to the spectra.
    - x (array-like): The spectra data to be plotted.
    - y (array-like): The associated measured proportions for color-coding.
    - axis_title (str): Title for the colorbar representing the associated measured proportions.
    - sample_ids (list, optional): List of sample IDs. If provided, a legend will be displayed.

    Returns:
    None
    """
    font = 'Arial'
    axis_label_size = 18
    tick_label_size = 11.43
    cmap = colormaps.get_cmap('jet')
    normalized_y = normalize_array(y)
    
    for idx, (spectrum, normalized_value) in enumerate(zip(x, normalized_y)):
        color = cmap(normalized_value)
        ax.plot(wavelengths, spectrum, c=color)
    
    if sample_ids is not None:
        ax.legend([f'{a} - {b:.2f}' for a, b in zip(sample_ids, y)], prop={'family': font, 'size': tick_label_size})
    
    ax.set_xlabel('')  # Remove the x-axis label for inner plots
    ax.set_ylabel('')  # Remove the y-axis label for inner plots
    if is_first_col:
        ax.set_ylabel('Intensity (a.u.)', fontsize=axis_label_size, fontname=font)  # Keep y-axis label for the first column
    if idx >= 3:
        ax.set_xlabel('Wavelength (nm)', fontsize=axis_label_size, fontname=font)  # Keep x-axis label for the last row
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.min(y), vmax=np.max(y)))
    sm.set_array([])  # This step is added to handle the warning
    cbar = plt.colorbar(sm, label=axis_title, pad=0.005, ax=ax)  # Use ax parameter to avoid warning
    cbar.ax.tick_params(labelsize=axis_label_size)  # Set colorbar tick label size
    cbar.set_label(axis_title, fontsize=axis_label_size, fontname=font)  # Set colorbar label size and font family

def Smoothing_Savitzky_Golay(specs, window_length=15, poly_order=2):
    """
    Apply Savitzky-Golay filter to each spectrum in the given data

    Parameters:
        specs (numpy.ndarray): 2D array of spectra where each row represents a spectrum.
        window_length (int): The length of the filter window. Default is 15.
        poly_order (int): The order of the polynomial used in filtering. Default is 2.
    """
    specs_filtered = np.zeros(specs.shape)
    
    for spectrum in range(specs.shape[0]):
        specs_filtered[spectrum, :] = savgol_filter(specs[spectrum, :], window_length, poly_order)
    return specs_filtered


