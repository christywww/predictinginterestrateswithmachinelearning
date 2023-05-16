import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.neural_network import MLPRegressor
from LSTMmodel import LSTMModel
from datetime import datetime
from sklearn import preprocessing
import seaborn as sns

class Model:
    def __init__(self, modeltype="SVM", text_df=None, bondrate_df=None, nr_topics=30, **kwargs):
        if text_df is None or bondrate_df is None:
            print('Parameters text_df and bondrate_df required!')

        self.text_df = text_df
        self.bondrate_df = bondrate_df
        self.nr_topics = nr_topics

        self.x = [str(x) + '_sentiment' for x in range(30)] + [str(x) + '_wordcount' for x in range(30)]
        self.y = ['15d change']

        self.modeltype = modeltype

        if modeltype == "SVM":
            self.model = svm.SVR(**kwargs)
        elif modeltype == "MLP":
            self.model = MLPRegressor(hidden_layer_sizes=(100,100,100), **kwargs)
        elif modeltype == "LSTM":
            self.model = LSTMModel(**kwargs)
        else:
            print("Model must be one of 'SVM', 'MLP', or 'LSTM'")

        if 'time_steps' in kwargs:
            self.time_steps = kwargs['time_steps']
        else:
            self.time_steps = None

    def merge_split_data(self):
        # convert both indices to string so that they join successfully (even if one is imported from csv and loses their datetime index)
        self.bondrate_df.index = self.bondrate_df.index.astype(str)
        self.text_df.index = self.text_df.index.astype(str)

        self.merged_df = self.bondrate_df.join(self.text_df, how="inner", sort=True, validate="one_to_one")

        if self.modeltype == "LSTM":
            self.model.process_data(self.merged_df, self.x, self.y)
            if self.time_steps is not None:
                self.merged_df = self.merged_df[self.time_steps:]
            else:
                self.merged_df = self.merged_df[10:]

        total_rows = self.merged_df.shape[0]
        self.x_train, self.y_train = self.merged_df[self.x][:int(total_rows * 0.85)], self.merged_df[self.y][:int(total_rows * 0.85)]
        self.x_test, self.y_test = self.merged_df[self.x][int(total_rows * 0.85):], self.merged_df[self.y][int(total_rows * 0.85):]

        print(datetime.now(), "split data into train and test sets")

    def preproccess_data(self):
        # remove / flag irrelevant topics (wordcount = 0 for past 30 meeting minutes)
        removed_topics = []
        for topic in range(self.nr_topics):
            if all([x == 0 for x in self.text_df[str(topic) + "_wordcount"].values[-30:]]):
                self.x.remove(str(topic) + "_sentiment")
                self.x.remove(str(topic) + "_wordcount")
                removed_topics.append(str(topic))
        print(datetime.now(), "Removed topics", ", ".join(removed_topics), "as they talk about historical events / are irrelevant now.")

        # normalization!
        for topic in range(self.nr_topics):
            scaler = MinMaxScaler()
            self.text_df[str(topic) + "_wordcount"] = scaler.fit_transform(self.text_df[[str(topic) + "_wordcount"]])
        print()

    def train(self):
        print(datetime.now(), "fitting", self.modeltype, "model...")
        self.model.fit(self.x_train, np.ravel(self.y_train))
        print(datetime.now(), self.modeltype, "model fitted.")

    def get_metrics(self):
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error
        from sklearn.metrics import confusion_matrix

        def change_to_direction(num): # utility function for confusion matrix
            if abs(num) < 0.01: # bond rate unchanged
                return 0
            elif num < 0: # bond rate decrease
                return -1
            else: # bond rate increases
                return 1

        # train set
        if self.modeltype == "LSTM":
            y_pred_train = self.model.predict(self.x_train, traintest="train")
        else:
            y_pred_train = self.model.predict(self.x_train)
        self.train_df = self.y_train.copy()
        self.train_df['train_pred'] = y_pred_train
        self.train_df.sort_index(inplace=True)
        print("=========================\n", self.modeltype, "train set:")
        print("MAE: ", mean_absolute_error(self.train_df["15d change"], self.train_df['train_pred']))
        print("MSE: ", mean_squared_error(self.train_df["15d change"], self.train_df['train_pred']))
        print("Confusion Matrix: \n", confusion_matrix([change_to_direction(val) for val in self.train_df["15d change"]], [change_to_direction(val) for val in self.train_df["train_pred"]]))

        # test set
        y_pred_test = self.model.predict(self.x_test)
        self.test_df = self.y_test.copy()
        self.test_df['test_pred'] = y_pred_test
        self.test_df.sort_index(inplace=True)
        print("=========================\n", self.modeltype, "test set:")
        print("MAE: ", mean_absolute_error(self.test_df["15d change"], self.test_df['test_pred']))
        print("MSE: ", mean_squared_error(self.test_df["15d change"], self.test_df['test_pred']))
        print("Confusion Matrix: \n", confusion_matrix([change_to_direction(val) for val in self.test_df["15d change"]], [change_to_direction(val) for val in self.test_df["test_pred"]]))

        # sns.lineplot(data=self.test_df)
        print('')


    def process(self):
        self.preproccess_data()
        self.merge_split_data()
        self.train()
        self.get_metrics()

if __name__ == "__main__":
    text_df = pd.read_csv('feature_df.csv', index_col=[0])
    bondrate_df = pd.read_csv('bondrate_df.csv', index_col=[0])

    # args = {"activation": "identity", "solver": "adam"}
    args = {"kernel": "rbf"}
    # args = {"time_steps": 30, "batch_size": 8}
    model = Model(modeltype="SVM", text_df=text_df, bondrate_df=bondrate_df, **args)
    model.process()
    print('')