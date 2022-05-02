import hdf5storage
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns

from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

from models import *

def add_mid_price_direction(df):
    """
    Takes the (scaled) dataframe with mid prices and 
    adds a column with induced mid price direction.
    """

    # add mid price change column
    mid_prices = df["Mid_Price"].to_list()
    price_directions = []
    for i in range(0, len(mid_prices) - 1):
        if mid_prices[i] == mid_prices[i + 1]:
            label = 'unchanged'
        elif mid_prices[i] < mid_prices[i + 1]:
            label = 'increase'
        else:
            label = 'decrease'
        price_directions.append(label)
    
    # manually pad with a final unchanged to match sizes
    price_directions.append('unchanged')

    df['Mid_Price_Change'] = price_directions

    return df

def data_spacing(df, step_size=250):
    """
    Takes in the cleaned dataframe and turns the 
    unevenly spaced timeseries in an evenly spaced timeseries.
    """

    # add column with mid price direction
    df = add_mid_price_direction(df)

    # This can be turned on if you potentially want to decrease class imbalaned
    # # remove 60% of instances without mid price change to decrease class imbalance
    # df = df.drop(df[df['Mid_Price_Change'] == 'unchanged'].sample(frac=.6).index)
    
    uneven_data = df.to_numpy()
    start_time = uneven_data[0, 0]
    end_time = uneven_data[-1, 0]

    # set the time scale we want the new df to have
    time_scale = np.arange(start_time, end_time, step_size)
    total_time = time_scale.shape[0]
    
    # create empty array of appropriate shape for even spaced data
    # subtract 2 columns for the time and change direction
    even_data = np.empty([total_time, uneven_data.shape[1] - 2])
    
    # find the data closest to the desired time stamp
    for i, time_stamp in enumerate(time_scale):
        if i % 1000 == 0:
            print(f'{i}/{total_time}')
        row = 0
        max_row = int(uneven_data.shape[0])
        while row < max_row and uneven_data[row, 0] < time_stamp:
            row += 1
        # add the appropriate row to the data for the timestamp 
        # (do not include time and change direction)    
        even_data[i] = uneven_data[row - 1, 1:-1]
    
    # concat the new time scale to the obtained data
    out_data = np.concatenate([time_scale.reshape(-1, 1), even_data], axis=1)

    return out_data

def remove_rows_0_midprice(df):
    df = df[df['Mid_Price'] != 0]
    return df

def remove_nan_inf(df):
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    return df

def remove_unreliable(df):
    df_copy = df.copy()
    size = len(df_copy)
    for col in df_copy.columns:
        perc = df_copy[col].isnull().sum() / float(size)

        # consider more than 40% NaN unreliable features
        if perc > 0.4:
            print('Deleting column {0}: missing perc={1:.2f}'.format(col, perc))    
            df_copy = df_copy.drop([col], axis=1)
            if col in df_copy.columns: 
                df_copy.columns.remove(col)

    return df_copy

def data_loader(data_path, columns, pickle_path, scaled_data=True):
    data = hdf5storage.loadmat(data_path)
    df = pd.DataFrame.from_dict(data['LOB'])
    df.columns = columns
    df = remove_rows_0_midprice(df)
    df = remove_unreliable(df)
    df = remove_nan_inf(df)
    if scaled_data:
        if pickle_path != None:
            try:
                df = pd.read_pickle(pickle_path)
            except:
                pickle_path = input("File not found, please check the path and \
try again: ")
                df = pd.read_pickle(pickle_path)
        else:
            spaced_data = data_spacing(df)
            df = pd.DataFrame(spaced_data)
        df.to_pickle('data/scaled_data_frame.pkl')
    df.columns = columns
    return df

def add_rolling_mean(df):
    """
    Adds a rolling mean column to the dataframe 
    to include mid price history.
    """
    df["rolling_mean"] = df['Mid_Price'].rolling(window=4).mean()

    return df

def shift_data(df):
    """
    Shifts all features with respect to the
    mid price to align target price and features.
    """
    # split time/mid price and features/classification label
    cols_features = df.columns[2:]
    df_shifted = df[["Mid_Price"]]

    # shift all features one snapshot to predict future snapshot values
    df_shifted = df_shifted.assign(**{
            f'{col}_(t-{1})': df[col].shift(1) for col in cols_features
            })

    return df_shifted

def variance_threshold_selection(
    Xtrn,
    Xtst, 
    variance_threshold,
    normalise=True
    ):
    """
    params:
    feature_matrix: 2d numpy array of features
        feature matrix has to be the training data only,
        in order to avoid test data leakage into variance calculation
    variance_threshold: float 

    returns:
    mask: 1d numpy array with booleans
        mask can be used to slice both train and test data for 
        feature selection.
    """

    # scale the data if scaling is switched on
    if normalise:
        scaler = MinMaxScaler()
        Xtrn = scaler.fit_transform(Xtrn)
    
    # calculate feature variance
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(Xtrn)
    mask = selector.get_support()

    # remove features with variance below threshold
    Xtrn_selected = Xtrn[:, mask]
    Xtst_selected = Xtst[:, mask]
    
    return Xtrn_selected, Xtst_selected

def standardise_data(Xtrn, Xtst):
    trn_scaler = StandardScaler().fit(Xtrn)
    tst_scaler = StandardScaler().fit(Xtst)
    Xtrn_s = trn_scaler.transform(Xtrn) # standardised training data
    Xtst_s = tst_scaler.transform(Xtst) # standardised test data

    return Xtrn_s, Xtst_s

def pca_dimension_reduction(Xtrn, Xtst, explained_var_threshold, standardise=True):
    """
    Takes in the training and test data, and executes the
    feature selection and engineering based on PCA.
    """

    if standardise:
        Xtrn_s, Xtst_s = standardise_data(Xtrn, Xtst)
    
    number_of_features = Xtrn_s.shape[1]

    for component_count in range(2, number_of_features):
        # calculate eigen vectors of standardised training data
        pca = PCA(n_components=component_count)
        pca.fit(Xtrn_s)

        # project both standardised train and test data
        Xtrn_s_reduced = pca.transform(Xtrn_s)
        Xtst_s_reduced = pca.transform(Xtst_s)
        explained_variance_ratios = pca.explained_variance_ratio_
        total_explained_variance_ratio = sum(list(explained_variance_ratios))

        # finish loop once minimal variance is reached
        if total_explained_variance_ratio > explained_var_threshold:
            return Xtrn_s_reduced, Xtst_s_reduced
    
def train_test_split(df, train_size):
    """
    Splits the entire data in a train and test split
    which can be used for final model performance check
    and initial experimentation.

    params:
    df: dataframe
        including target price and classification labels
    train_size: float
        ratio that determines the size of the train and test data splits

    returns:
    df_trn, df_tst: dataframes
        two dataframes, one for train data and one for test data
    """

    total_instances = len(df)
    split_index = int(train_size * total_instances)
    df_trn = df.iloc[:split_index, :]
    df_tst = df.iloc[split_index:, :]

    return df_trn, df_tst

def split_target_features_labels(df):
    """
    Turns a dataframe with regression targets, features 
    and classification labels into separate numpy arrays.
    """
    # future mid price (regression targets)
    target_price = np.asarray(df["Mid_Price"])
    # all features from ask1 (leave out mid price, spread and labels)
    feature_matrix = np.asarray(df.iloc[:, 2: -1])
    # induced price change of feature row (decreased, unchanged, increased)
    classification_labels = np.asarray(df["Mid_Price_Change_(t-1)"])

    return feature_matrix, target_price, classification_labels

def two_d_feature_generation(X, Y, look_back):
    """
    Takes the time series and transforms 
    them into the 3D input of the format:
    [samples, lookback period, features]
    """
    x_train_append_matrix = []
    y_train_append_matrix = []

    for i in range(len(X)-look_back):         
        feat_current_train = X[i:i+look_back, :]
        label_current_train = Y[i+look_back]
        x_train_append_matrix.append(feat_current_train)
        y_train_append_matrix.append(label_current_train)
    
    X_numpy = np.array(x_train_append_matrix)
    Y_numpy = np.array(y_train_append_matrix)

    return X_numpy, Y_numpy

def feature_reduction_selection(
                    Xtrn, 
                    Xtst, 
                    selection, 
                    reduction, 
                    var_thres=0.001, 
                    var_ratio_thres=0.9
    ):
    """
    Takes the train and test features, and applies the selected
    reduction and selection method for the configuration.
    """
    if selection == 'var_threshold':
        Xtrn, Xtst = variance_threshold_selection(Xtrn, Xtst, var_thres)
    if reduction == 'PCA':
        Xtrn, Xtst = pca_dimension_reduction(Xtrn, Xtst, var_ratio_thres)

    return Xtrn, Xtst

def predict_true_scatter(
            Y_tst, 
            Y_tst_pred, 
            model, 
            selection, 
            reduction,
            output_path
    ):
    """
    Takes in the confirguration, original target and predicted tartgets
    and plots the true/predicted regression with a y=x line in one figure.
    """
    all_mid_prices = set(Y_tst.flatten()) | set(Y_tst_pred.flatten())
    smallest_mid_price = int(min(all_mid_prices))
    largest_mid_price = int(max(all_mid_prices))

    smallest_true_mid_price = int(min(Y_tst))
    largest_true_mid_price = int(max(Y_tst))

    # to avoid a memory error, skip the plotting of really bad predictions
    if largest_mid_price > 10000000:
        return

    # set ranges and ticks for the plots
    x_range = [smallest_true_mid_price, largest_true_mid_price]
    plot_window = [smallest_mid_price, largest_mid_price]
    tick_interval = int((largest_mid_price - smallest_mid_price) / 4)
    ticks = list(range(plot_window[0], plot_window[1], tick_interval))
    diagonal_x = diagonal_y = list(range(plot_window[0], plot_window[1], 1))
    file_title = f'{model}_{selection}_{reduction}.png'

    # build the plot
    fig = plt.figure(figsize=(10,10), dpi=100)
    ax = fig.add_subplot(111)
    ax.scatter(Y_tst, Y_tst_pred)
    ax.scatter(diagonal_x, diagonal_y, s=1)
    ax.set_xlabel("True mid price", fontsize=15)
    ax.set_ylabel("Predicted mid price", fontsize=15)
    if reduction == "PCA":
        ax.set_xlim(x_range)
        ax.set_ylim(plot_window)
    else:
        ax.set_xlim(plot_window)
        ax.set_ylim(plot_window)
        ax.set_xticks(ticks, fontsize=10)
        ax.set_yticks(ticks, fontsize=10)
    fig.savefig(os.path.join(output_path, 'figures', file_title))

    return

def sort_models_by_score(score_dict, output_path):
    """
    Takes the calculated scores of all configurations for all folds,
    calculates mean and standard deviation and sorts the scores.
    """

    scores_per_model = {}
    for model in score_dict.keys():
        for selection_method in score_dict[model].keys():
            for reduction_method in score_dict[model][selection_method].keys():
                score_arr = np.asarray(
                    score_dict[model][selection_method][reduction_method]
                )
                mean = np.mean(score_arr)
                std = np.std(score_arr)
                scores_per_model[
                    f'{model}_{selection_method}_{reduction_method}'
                ] = (mean, std)
    
    sorted_by_mean = {
                    k: {'mean': v[0], 'std': v[1]} for k,v
                    in sorted(scores_per_model.items(), key=lambda x: x[1][0])
    }
    
    with open(output_path, 'w') as f:
        json.dump(sorted_by_mean, f)

    return

def regression_experiment(
            data_path, 
            models, 
            selection_methods,
            reduction_methods,
            fig_output_path,
            columns,
            pickle_path,
            scaled_data=True,
            cnn_look_back=10
    ):
    """
    Takes in all the possible configurations, and does a grid search for
    the best MSE over all configuration combinations.
    """

    df = data_loader(data_path, columns, pickle_path, scaled_data)
    df = add_rolling_mean(df)
    df = add_mid_price_direction(df)
    df = shift_data(df)

    # remove first 4 lines with NaN values left from pre-processing
    df = df.iloc[4:, :]

    # split features, targets and labels
    X, Y, classification_labels = split_target_features_labels(df)

    # initialize the time series fold indexes 
    tscv = TimeSeriesSplit()

    # initialize the test score dict
    test_scores = {model: {} for model in models.keys()}

    # TODO: turn nested loops into itertool combination
    # run all predictions and add scores to dictionary
    for model in models.keys():
        for selection in selection_methods:
            for reduction in reduction_methods:
                fold_test_scores = []
                print(f'doing folds for {model} | {selection} | {reduction}')
                # interate over all folds and store the scores 
                # of each configuration as an array
                fold_count = 1
                for train_index_array, test_index_array in tscv.split(X):
                    # select train and test folds
                    Xtrn, Xtst = X[train_index_array, :], X[test_index_array, :]
                    Ytrn, Ytst = Y[train_index_array], Y[test_index_array]

                    # feature reduction/selection
                    Xtrn, Xtst = feature_reduction_selection(
                                                        Xtrn, 
                                                        Xtst, 
                                                        selection, 
                                                        reduction
                    )

                    # cnn feature transformation and training/prediction
                    if model == 'cnn' or model == 'lstm':
                        Xtrn_cnn, Ytrn_cnn = two_d_feature_generation(
                                                                Xtrn, 
                                                                Ytrn, 
                                                                cnn_look_back
                        )
                        Xtst_cnn, Ytst_cnn = two_d_feature_generation(
                                                                Xtst, 
                                                                Ytst, 
                                                                cnn_look_back
                        )
                        trained_model = models[model](
                                                Xtrn_cnn, 
                                                Ytrn_cnn, 
                                                cnn_look_back
                        )
                        Ytst_pred = trained_model.predict(Xtst_cnn)
                        test_score = mean_squared_error(Ytst_cnn, Ytst_pred)
                        if fold_count == 5 and selection != "var_threshold":
                            predict_true_scatter(
                                            Ytst_cnn, 
                                            Ytst_pred, 
                                            model, 
                                            selection, 
                                            reduction,
                                            fig_output_path
                            )
                    # non-cnn training/prediction
                    else:
                        trained_model = models[model](Xtrn, Ytrn)
                        Ytst_pred = trained_model.predict(Xtst)
                        if reduction == "PCA":
                            print(Ytst)
                            print(Ytst_pred)
                        test_score = mean_squared_error(Ytst, Ytst_pred)
                        if fold_count == 5 and selection != "var_threshold":
                            predict_true_scatter(
                                            Ytst, 
                                            Ytst_pred, 
                                            model, 
                                            selection, 
                                            reduction,
                                            fig_output_path
                            )
                    fold_test_scores.append(test_score)
                    fold_count += 1

                # store MSE score on test data
                # nested dictionaries for all combinations of configurations
                if selection not in test_scores[model]:
                    test_scores[model][selection] = {reduction: fold_test_scores}
                else: 
                    test_scores[model][selection][reduction] = fold_test_scores

    # dump raw score arrays
    with open(os.path.join(fig_output_path, "raw_scores.json"), 'w') as f:
        json.dump(test_scores, f)      
    return test_scores

def classification_experiment(data_path, fig_output_path, columns, pickle_path):
    """
    Does the classification experiment on the data. Only one configuration 
    is explored and this requires further testing/pre processing.
    """

    # pre_processing
    df = data_loader(data_path, columns, pickle_path, False)
    df = add_rolling_mean(df)
    df = add_mid_price_direction(df)
    df = shift_data(df) # this is only necessary because of a code inflexibility
                        # which shows in split_target_features and has no actual
                        # use. The reason is the column naming: _(t-1)

    # remove first 4 lines with NaN values left from pre-processing
    df = df.iloc[4:, :]

    # split features, targets and labels
    X, Y, classification_labels = split_target_features_labels(df)

    tscv = TimeSeriesSplit()
    accuracies = []
    for train_index_array, test_index_array in tscv.split(X):
        # select train and test folds
        Xtrn, Xtst = (
                X[train_index_array, :], 
                X[test_index_array, :]
        )
        Ytrn, Ytst = (
                classification_labels[train_index_array], 
                classification_labels[test_index_array]
        )

        # encode the output labels
        encoder = LabelEncoder()
        encoder.fit(Ytrn)
        encoded_Y_trn = encoder.transform(Ytrn)
        encoded_Y_tst = encoder.transform(Ytst)

        # one hot encode the output labels
        dummy_Y_trn = (
            np.asarray(np_utils.to_categorical(encoded_Y_trn)).astype(np.float32)
        )
        dummy_Y_tst = (
            np.asarray(np_utils.to_categorical(encoded_Y_tst)).astype(np.float32)
        )

        # model training
        model = classification_model(Xtrn, dummy_Y_trn)

        # model evaluation
        test_loss, test_acc = model.evaluate(Xtst, dummy_Y_tst, verbose=2)
        accuracies.append(test_acc)

    with open('output/original/accuracies.json', 'w') as f:
        json.dump(accuracies, f)

    plt.figure()
    sns.histplot(data=df, x='Mid_Price_Change_(t-1)')
    plt.savefig('output/original/class_balance.png')

    return 

if __name__ == "__main__":
    # global variable definitions
    LOB_COLUMN_NAMES = [
        "Time", "Mid_Price", "Spread", 
        "AskPrice1", "AskVolume1","BidPrice1", "BidVolume1", # Level 1
        "AskPrice2", "AskVolume2","BidPrice2", "BidVolume2", # Level 2
        "AskPrice3", "AskVolume3","BidPrice3", "BidVolume3", # Level 3
        "AskPrice4", "AskVolume4","BidPrice4", "BidVolume4", # Level 4
        "AskPrice5", "AskVolume5","BidPrice5", "BidVolume5", # Level 5
        "AskPrice6", "AskVolume6","BidPrice6", "BidVolume6", # Level 6
        "AskPrice7", "AskVolume7","BidPrice7", "BidVolume7", # Level 7
        "AskPrice8", "AskVolume8","BidPrice8", "BidVolume8", # Level 8
        "AskPrice9", "AskVolume9","BidPrice9", "BidVolume9", # Level 9
        "AskPrice10", "AskVolume10","BidPrice10", "BidVolume10", # Level 10
    ]
    DATA_DIR = 'data'
    FIG_OUTPUT_SCALED_DIR = 'output/scaled'
    FIG_OUTPUT_ORIGINAL_DIR = 'output/original'
    RAW_LOB_DATA = "S092215-v50-UUID_OCT2_states.mat"
    #"S092215-v50-AMZN_OCT2_states.mat"

    # can add lstm here, but is omitted because of slow training times
    MODELS = {
        'mlp': shallow_mlp, 
        'deep_mlp': deep_mlp,
        'deepest_mlp': deepest_mlp, 
        'cnn': cnn_regression,
    }
    SELECTION_METHODS = ['none', 'var_threshold']
    REDUCTION_METHODS = ['none', 'PCA']

    # some user input
    pickle_path = None
    scaled_answer = input("WARNING: The process of scaling the uneven \
timeseries is time consuming.\n\n\
Do you already have a pickled scaled dataframe? (y/n + enter) ")
    if scaled_answer.lower() == 'y':
        pickle_path = input("Please provide the path to the pickled data \
relative to the Python interpreter: ")

    #------- this is the regression experiment with scaling -------
    scaled_test_scores = regression_experiment(
                            os.path.join(DATA_DIR, RAW_LOB_DATA), 
                            MODELS, 
                            SELECTION_METHODS, 
                            REDUCTION_METHODS, 
                            FIG_OUTPUT_SCALED_DIR,
                            LOB_COLUMN_NAMES,
                            pickle_path
    )
    # sort the models by mean k-fold scores
    sort_models_by_score(
                    scaled_test_scores, 
                    os.path.join(FIG_OUTPUT_SCALED_DIR,
                    "sorted_scores_scaled.json")
    )
    
    #------- this is the regression experiment with original data -------
    original_test_scores = regression_experiment(
              os.path.join(DATA_DIR, RAW_LOB_DATA), 
              MODELS, SELECTION_METHODS, 
              REDUCTION_METHODS, 
              FIG_OUTPUT_ORIGINAL_DIR,
              LOB_COLUMN_NAMES,
              pickle_path,
              scaled_data=False,
    )
    # sort the models by mean k-fold scores
    sort_models_by_score(
                    original_test_scores, 
                    os.path.join(FIG_OUTPUT_ORIGINAL_DIR,
                    "sorted_scores_original.json")
    )
    
    #------- this is the classification experiment with original data -------
    classification_experiment(
                    os.path.join(DATA_DIR, RAW_LOB_DATA), 
                    FIG_OUTPUT_ORIGINAL_DIR, 
                    LOB_COLUMN_NAMES,
                    pickle_path
    )

