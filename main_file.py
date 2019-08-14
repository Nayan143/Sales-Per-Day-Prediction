import time
import keras
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from keras.models import Sequential
from keras.layers import Dense, Dropout
from joblib import dump

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        logging.info("Time taken = {}".format(te-ts))
        return result
    return timed


class MultiColumnLabelEncoder:
    def __init__(self, columns = None):
        # array of column names to encode
        self.columns = columns

    def fit(self, X, y=None):
        # not relevant here
        return self

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''

        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class MMSRegression:

    def __init__(self):
        self.mean_absolute_deviations = np.inf

    def dispose(self):
        """
        Deletes this instance from memory
        :return:
        """
        del (self)

    def get_scores(self, y_true, y_pred, x_test):
        """
        This function calculates some popular metrics for regression by
        comparing the true and predicted values
        :param y_true: the ground true reference
        :param y_pred: the predictions
        :param x_test: this has been kept only to count the total number of features
        :return: None
        """
        try:
            assert (y_true.shape[0] == y_pred.shape[0])
            print("RMSE {}".format(np.sqrt(mse(y_true, y_pred))))
            mean_absolute_deviations = mae(y_true, y_pred)
            print("MAE {}".format(mean_absolute_deviations))
            self.mean_absolute_deviations = mean_absolute_deviations
        except AssertionError as error:
            logging.error("Unequal number of observations")

    def get_data(self, path):
        """
        This function reads data from disk
        :param path: A path on the disk
        :return: raw data of type 2d pandas arrays.
        """
        try:
            raw_data = pd.read_csv(path, delimiter=';')
            return raw_data
        except:
            logging.error("Invalid path")

    def scaling_operation(self, do_scaling, X, y):

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

            if do_scaling:
                scaler_x_minmax = MinMaxScaler()

                scaler_x_minmax.fit(X_train)
                X_train_scaled = scaler_x_minmax.transform(X_train)

                X_test_scaled = scaler_x_minmax.transform(X_test)

                scaler_y_minmax = MinMaxScaler()
                scaler_y_minmax.fit(y_train)
                y_train_scaled = scaler_y_minmax.transform(y_train)

                y_test_scaled = scaler_y_minmax.transform(y_test)

                return scaler_y_minmax, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_x_minmax

            else:

                return None, X_train, X_test, y_train, y_test, None
        except:
            logging.error("Something went wrong...")


    def get_plots(self, y_true, y_pred, x_tick_labels):
        """
        This function plot the predictions against true data points
        :param y_true: true values
        :param y_pred: predictions
        :param x_tick_labels: markers for x-axis
        :return: None
        """
        try:
            assert (y_true.shape[0] == y_pred.shape[0])
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            xc = np.arange(len(x_tick_labels))
            ax1.plot(xc, y_pred, label='pred')
            ax1.plot(xc, y_true, label='true')
            ax1.set_ylabel("Sales Per Day")
            plt.legend()
            plt.show()

        except AssertionError as error:
            logging.error("Unequal number of samples in output")



    def pre_process(self, data):
        """
        this function is responsible to prepare the data set for training and testing
        :param data: the raw data
        :return: normalized training input and output, scaling object of the output data and a series of instances
        """

        try:
            brand = data.iloc[:, 0].astype(str)
            country = data.iloc[:, 1].astype(str)
            currency = data.iloc[:, 2].astype(str)
            # replace NaN values with zero
            customers_per_day = data.iloc[:, 3].fillna(0)
            # replace NaN values with zero
            outlet_id = data.iloc[:, 4].fillna(0)
            # replace NaN values with zero
            week_id = data.iloc[:, 5].fillna(0)
            weekday = data.iloc[:, 6].astype(str)
            # replace NaN values with zero
            sales_per_day = data.iloc[:, 7].fillna(0)

            # stack independent and dependent feature variables
            processed_data = pd.concat([brand, country, currency, customers_per_day,
                                        outlet_id, week_id, weekday, sales_per_day], axis=1)
            # processed_data = pd.DataFrame(processed_data)

            # Drop all the negative values from sales_per_day and customers_per_day column
            processed_data = processed_data[(processed_data['sales_per_day'] >= 0) & (processed_data['customers_per_day'] >= 0)]

            # data with label encoding (get the unique number for each unique observation)
            encodedData = MultiColumnLabelEncoder(
                columns=['brand', 'country', 'currency', 'weekday']).fit_transform(processed_data)

            # getting all the raw data features
            #raw_data_features = encodedData.iloc[:, 0:-1].values
            # getting the target value
            #raw_data_target = encodedData.iloc[:, 7].values.reshape(-1, 1)

            # mutual information graph
            # mutual_info_regr(raw_data_features, raw_data_target)

            # Choosing the best features based on mutual information graph
            best_feature_1 = encodedData.iloc[:, 3]
            best_feature_2 = encodedData.iloc[:, 6]

            train_data_X = pd.concat([best_feature_1, best_feature_2], axis=1)
            target_data_y = raw_data_target

            # set the scaling true (scaled data) or false (processed raw data)
            scaling = True

            # all features and target scaled/raw (Scaled -> True/False) data
            scaler_y_minmax, X_train_scaled, X_test_scaled, \
            y_train_scaled, y_test_scaled, scaler_x_minmax = self.scaling_operation(scaling, train_data_X,
                                                                               target_data_y)

            # test instances are saved for latter plotting purposes
            test_instances = X_test_scaled[:, 0]

            return scaler_y_minmax, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_x_minmax, test_instances

        except AssertionError as error:
            print("input data cannot have NaN or Inf values")

    def train(self, X, y, use_keras=False, params=None):
        """
        initiating and fitting ML model
        :param X: training input
        :param y: training output
        :param use_keras: True (Scaled Data)/ False (raw data)
        :param params: trained model
        :return:
        """
        try:
            assert (X.shape[0] == y.shape[0])
            logging.info("Training model")
            if params == None:

                num_layers = 2
                num_neurons = 130
                activation = 'relu'
                learning_rate = 1e-4
                n_epochs = 10
                batch_size = 64
                dropout = Dropout(0.2)

            else:
                num_layers = params['num_layers']
                num_neurons = params['num_neurons']
                activation = params['activation']
                learning_rate = params['learning_rate']
                n_epochs = params['n_epochs']
                batch_size = params['batch_size']
                dropout = params['dropout']

            if use_keras:

                keras.backend.clear_session()

                # Choose an Optimizer
                optimizer = keras.optimizers.Adam(lr=learning_rate)

                # Initialize a sequential / feed forward model
                model = Sequential()

                # Add input and first hidden layer
                model.add(Dense(units=num_neurons, activation=activation, input_dim=X.shape[1]))

                # add dropout
                model.add(dropout)

                # Add subsequent hidden layer
                for _ in range(num_layers - 1):
                    model.add(Dense(units=num_neurons,
                                        activation=activation
                                        )
                                  )

                # Add Output Layer
                model.add(Dense(units=y.shape[1], activation='relu'))

                # Compile the regressor
                model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])

                history = model.fit(X, y, validation_split=0.20,
                                        epochs=n_epochs, batch_size=batch_size,
                                        verbose=1, shuffle=True)

                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('Model loss')
                plt.ylabel('Loss (mae)')
                plt.xlabel('Number of epochs')
                plt.legend(['training', 'validation'], loc='upper right')
                plt.grid()
                plt.show()

            else:

                model = DecisionTreeRegressor()
                model.fit(X, y)
            return model

        except AssertionError as error:
            logging.error("Unequal number of samples")

    def predict(self, model, input):
        """
        This function gives predictions for a certain input
        :param model: trained model
        :param input: (test) input
        :return: predictions
        """
        logging.info("Predicting")
        output = model.predict(input)

        return output

    def main(self, path):
        """
        this is the main function that initiates the pipeline
        :param path: the path on the disk where the data is kept
        :return: None
        """
        try:
            assert (path != "")
            logging.info("Starting pipeline")
            data = self.get_data(path)

            # checking if the data has been imported correctly
            logging.info("Shape of data imported: " + str(data.shape))

            # pre processing the data
            scaler_y_minmax, X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_x_minmax, test_instances = self.pre_process(data)

            # Hyper parameter tuning for neural network
            param = {}
            param['num_layers'] = 4
            param['num_neurons'] = 130
            param['activation'] = 'relu'
            param['learning_rate'] = 0.01
            param['n_epochs'] = 10
            param['batch_size'] = 64
            param['dropout'] = Dropout(0.3)

            # training
            self.model = self.train(X_train_scaled, y_train_scaled, use_keras=True, params=param)

            # making predictions on the transformed dataset
            y_pred_scaled = self.predict(self.model, X_test_scaled)

            # inverting the predictions to their original scale
            #y_pred = self.post_process(y_pred_raw, scaler_y)

            # generating scores
            self.get_scores(y_test_scaled, y_pred_scaled, X_test_scaled)

            # persist model if model-accuracy is satisfactory
            if self.mean_absolute_deviations < 1:
                dump(self.model, "model.pkl")

            # generating plots
            self.get_plots(y_test_scaled[0:20], y_pred_scaled[0:20], test_instances[0:20])

            return None
        except AssertionError as error:

            logging.error("Path cannot be null")

if __name__ == "__main__":
    path = "data.csv"
    MMSRegression().main(path)
