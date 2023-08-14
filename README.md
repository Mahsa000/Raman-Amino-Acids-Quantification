# Machine Learning Analysis of Raman Spectra to Quantify the Organic Constituents in Complex Organic-Mineral Mixtures

Welcome to the Raman analysis repository! This repository contains code, data, and documentation related to the analysis of memetic organic-mineral soil composition using Raman spectroscopy. The project aims to overcome challenges posed by complex organic/mineral compositions and fluorescence interference in soil analysis. Here we aim to provide the best practices on how to boost the quantitative power of Raman spectroscopy as a probe of chemical composition in complex mixtures.

## Table of Contents

- [Codes](#codes)
- [Data](#data)
- [Predictions](#predictions)
- [Helper Functions](#helper-functions)
- [License](#license)
- [Abstract](#abstract)
- [Code Details](#code-details)

## Codes

Explore the following Jupyter notebooks for data analysis and model development:

- [Amino_Acids_AA.ipynb](Codes/Amino_Acids_AA.ipynb)
- [Amino_Acids_Fluorescence_AAF.ipynb](Codes/Amino_Acids_Fluorescence_AAF.ipynb)
- [Amino_Acids_Minerals_AAM.ipynb](Codes/Amino_Acids_Minerals_AAM.ipynb)
- [Supporting_Information_Extra.ipynb](Codes/Supporting_Information_Extra.ipynb)

## Data

Access datasets for various soil compositions and ground truth data:

- [AAF_Data](Data/AAF_Data)
- [AAM_Data](Data/AAM_Data)
- [AA_Data](Data/AA_Data)
- [AA_GroundTruth](Data/AA_GroundTruth)
- [AA_Pure](Data/AA_Pure)

## Predictions

Find predicted results from the models in the [Predictions](Predictions) directory.

## Helper Functions

Explore helpful Python scripts for data loading, preprocessing, and model creation:

- [load_data.py](Helper_Functions/load_data.py)
- [models.py](Helper_Functions/models.py)
- [utils.py](Helper_Functions/utils.py)

## License

The repository is licensed under [LICENSE](LICENSE).

## Abstract

Important decisions in local agricultural policy and practice often hinge on the soil’s chemical composition. Raman spectroscopy offers a rapid non-invasive means to quantify the constituents of complex organic systems. But the application of Raman spectroscopy to soils presents a multifaceted challenge due to organic/mineral compo- sitional complexity and spectral interference arising from overwhelming fluorescence. The present work compares methodologies with the capacity to help overcome com- mon obstacles that arise in the analysis of soils. We create conditions representative of these challenges by combining varying proportions of six amino acids commonly found in soils with fluorescent bentonite clay and coarse mineral components. Referring to an extensive dataset of Raman spectra, we compare the performance of the convolutional neural network (CNN) and partial least squares regression (PLSR) multivariate models for amino acid composition. Strategies employing volume-averaged spectral sampling and data preprocessing algorithms improve the predictive power of these models. Our average test R2 for PLSR models exceeds 0.89 and approaches 0.98, depending on the complexity of the matrix, whereas CNN yields an R2 range from 0.91 to 0.97, demon- strating that classic PLSR and CNN perform comparably, except in cases where the signal-to-noise ratio of the organic component is very low, whereupon CNN models outperform. Artificially isolating two of the most prevalent obstacles in evaluating the Raman spectra of soils, we have characterized the effect of each obstacle on the perfor- mance of machine learning models in the absence of other complexities. These results highlight important considerations and modeling strategies necessary to improve the Raman analysis of organic compounds in complex mixtures in the presence of mineral spectral components and significant fluorescence.

## Code Details

The Jupyter notebooks cover various aspects of the project, including data preprocessing and regression model development:

- **Preprocessing:** Implementation and comparison of pre-processing methods for PLSR and CNN input. Techniques include Iterative Discrete Wavelet Transform (IDWT) baseline correction and spectral normalization and averaging as well as DWT dimensionality reduction, Savitzky-Golay smoothing, and the effect of random noise.
- **Regression Models:** Development of PLSR and CNN models to predict six amino acid proportions from Raman spectra. These models are trained and evaluated using cross-validation techniques, with model performance measured using R^2, MSE, and MAE metrics.

For more details, refer to the provided code and data files and the associated paper.
