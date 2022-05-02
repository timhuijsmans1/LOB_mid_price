=========================================================================
Python script for forecasting mid prices from LOB data
Author: Tim Huijsmans
=========================================================================

Script and data README

=========================================================================

Running:
- Run $python experiment.py
- When prompted for the scaled data frame, press y or n and continue
- When prompted for the path to the scaled data, make sure to use the path relative to the Python - interpreter

Contents:
/data contains the input data and will be used to store the scaled df
/output contains the empty folder required for the figures being stored and the output JSONs
.models.py contains the ML models used for forecasting
.experiment.py contains all the helper functions for preprocessing/plotting and the experiments

Input:
The experiment presented in the script is ran for two differently pre-processed data frames. The first one being the original data, and the second one being the scaled time series. As prompted by the terminal warning, the time series scaling takes a considerable amount of time. Therefore, as this is done once, the scaled dataframe is pickled and will be stored in the data folder. On the next time running the script, this path can be used to read the scaled dataframe from disk and save time.

Output:
The submission folder contains various output files for the two types of pre-processed data. In each file, the raw output scores, the statistics on these scores, and the relevant plots will be stored as JSONs and PNGs respectively.
