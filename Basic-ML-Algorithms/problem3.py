# Some Classification algorithms

import sys
import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

CSV_EXIST_ERROR_MSG = "The given path is not a csv file"
ARGS_NUM_ERROR_MSG = "The number of arguments is not valid. Usage: python3 problem3.py <CSV file>"
POS_VAL = 1
ZERO_VAL = 0

class PrepareData:
    # class for preparing the data  
    def __init__(self, data_matrix):
        self.raw_data = data_matrix

    def split_data(self):
        k_fold = 5
        test_ratio = 0.4
        train_ratio = 0.6
        X = self.raw_data[:, 0:2]
        y = self.raw_data[:, 2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_ratio, 
                                                            test_size=test_ratio, stratify=y) # splitting stratified
        return X_test, y_test, X_train, y_train

class TrainTestModel:
    # class for training and testing the models
    def __init__(self, X_test, y_test, X_train, y_train, data_matrix):
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.raw_data = data_matrix

    def plot_the_confusion_matrix(self, cm):
        df_cm = pd.DataFrame(cm, range(2), range(2))
        sn.set(font_scale=1.4)
        sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, cmap=plt.cm.Blues, fmt='d')
        plt.title('Confusion matrix', {'color': 'green'})
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.show()

    def plot_decision_boundaries_example(self, clf):
        # an example how to do it - here only on Decision Trees algorithm
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
        cmap_bold  = ListedColormap(['#FF0000', '#00FF00'])
        X = self.raw_data[:, 0:2]
        y = self.raw_data[:, 2]
        clf.fit(X, y)
        mesh_step_size = .01  # step size in the mesh
        plot_symbol_size = 50
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, mesh_step_size), np.arange(y_min, y_max, mesh_step_size))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
        # Plot training points
        plt.scatter(X[:, 0], X[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor='black')
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        patch0 = mpatches.Patch(color='#FF0000', label='0')
        patch1 = mpatches.Patch(color='#00FF00', label='1')
        plt.legend(handles=[patch0, patch1], loc='lower right', bbox_to_anchor=(0.5, 1.1), ncol=3, 
                   fancybox=True, shadow=True)
        plt.xlabel('A')
        plt.ylabel('B')
        plt.title("Color map - decision boundaries")
        plt.show()

    def print_best_scores(self, grid_search, y_prediction):
        print("Best parameters: \n" + str(grid_search.best_params_))
        # print("Best accuracy: \n" + str(grid_search.best_score_))
        print("Accuracy score on prediction: \n" + str(accuracy_score(self.y_test, y_prediction)))
        labels = [False, True]
        cm = confusion_matrix(self.y_test, y_prediction, labels)
        print("Confusion Matrix: \n" + str(cm))
        self.plot_the_confusion_matrix(cm)
        print("--- Statistics ---")
        print('number of test examples (by classes): \n' +  str(pd.Series(self.y_test).value_counts()))
        print('distribution of the prediction (by classes): \n' + str(pd.Series(y_prediction).value_counts()))

    def train_model(self, classifier, tuned_parameters):
        k_fold = 5
        grid_search = GridSearchCV(estimator=classifier,
                                   param_grid=tuned_parameters,
                                   scoring='accuracy', # could be also: 'precision', 'recall', 'f1'
                                   cv=k_fold, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        y_prediction = grid_search.predict(self.X_test)
        self.print_best_scores(grid_search, y_prediction)
        return grid_search

    def train_svm_linear(self):
         # svm with grid-search for tuning the hyper-parameters:
        tuned_parameters = [{'kernel': ['linear'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]
        svc = svm.SVC()
        self.train_model(svc, tuned_parameters)

    def train_svm_polynomial(self):
        # svm with grid-search for tuning the hyper-parameters:
        tuned_parameters = [{'kernel': ['poly'], 'C': [0.1, 1, 3], 'degree': [4, 5, 6], 'gamma': [0.1, 0.5]}]
        svc = svm.SVC()
        self.train_model(svc, tuned_parameters)

    def train_svm_rbf(self):
        # svm with grid-search for tuning the hyper-parameters:
        tuned_parameters = [{'kernel': ['rbf'], 'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]}]
        svc = svm.SVC()
        self.train_model(svc, tuned_parameters)

    def train_logistic_regression(self):
        # logistic_regression with grid-search for tuning the hyper-parameters:
        tuned_parameters = [{'C': [0.1, 0.5, 1, 5, 10, 50, 100]}]
        lr = LogisticRegression()
        self.train_model(lr, tuned_parameters)

    def train_knn(self):
        # k-nearest neighbors with grid-search for tuning the hyper-parameters:
        tuned_parameters = [{'n_neighbors': [1, 10, 20, 30, 40 ,50], 'leaf_size': [5, 15, 25, 35, 45, 55, 60]}]
        knn = KNeighborsClassifier()
        self.train_model(knn, tuned_parameters)

    def train_decision_trees(self):
        # decision trees with grid-search for tuning the hyper-parameters:
        tuned_parameters = [{'max_depth': [1, 10, 20, 30, 40 ,50], 'min_samples_split': [2, 4, 6, 8, 10]}]
        dt = tree.DecisionTreeClassifier()
        # get the best parameters and plot the tree:
        trained_clf = self.train_model(dt, tuned_parameters)
        clf = tree.DecisionTreeClassifier(min_samples_split=trained_clf.best_params_["min_samples_split"], 
                                          max_depth=trained_clf.best_params_["max_depth"])
        # Plot the tree:
        # clf = clf.fit(self.X_train, self.y_train)
        # tree.plot_tree(clf)
        # Plot color map
        self.plot_decision_boundaries_example(clf)

    def train_random_forest(self):
        # random forest with grid-search for tuning the hyper-parameters:
        tuned_parameters = [{'max_depth': [1, 10, 20, 30, 40 ,50], 'min_samples_split': [2, 4, 6, 8, 10]}]
        rf = RandomForestClassifier()
        self.train_model(rf, tuned_parameters)

class DrawGraph:
    # class for drawing the model
    # assuming the data matrix has 2 features and 1 label
    def __init__(self, data_matrix):
        self.data_matrix = data_matrix[:, 0:2]
        self.labels = data_matrix[:, 2]
        self.labels = self.labels.reshape((self.labels.shape[0], 1))

    def split_pos_zero(self):
        pos_idxs = np.asarray(np.where(self.labels == POS_VAL))
        zero_idxs = np.asarray(np.where(self.labels == ZERO_VAL))
        pos_vector = self.data_matrix[pos_idxs[0]]
        zero_vector = self.data_matrix[zero_idxs[0]]
        return pos_vector, zero_vector

    def draw_dots(self):
        pos_vector, zero_vector = self.split_pos_zero()
        plt.scatter(pos_vector[:, 0], pos_vector[:, 1], color='blue', label='one')
        plt.scatter(zero_vector[:, 0], zero_vector[:, 1], color='green', label='zero')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, 
                   fancybox=True, shadow=True)

class ParseCSV:
    # class for reading csv files
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.check_for_erros()

    def check_for_erros(self):
        if (not path.exists(self.csv_path) or 
            not path.isfile(self.csv_path) or
            self.csv_path[-3:] != 'csv'):
            sys.exit(CSV_EXIST_ERROR_MSG)

    def read_csv(self):
        csv_file = pd.read_csv(self.csv_path, header=0)
        return csv_file

    def data_to_matrices(self, csv_file):
        csv_cols = list(csv_file.columns)
        final_matrix = np.zeros(csv_file.shape)
        for i in range(len(csv_cols)):
            final_matrix[:, i] = csv_file[csv_cols[i]].values
        return final_matrix

class Test:
    # class for tests
    def __init__(self):
        pass

    def test_print(self, text, data):
        print(text)
        print(data)
        print('----')

    def test_read_csv(self, csv_path):
        csv_obj = ParseCSV(csv_path)
        csv_file = csv_obj.read_csv()
        return csv_obj.data_to_matrices(csv_file)

    def test_draw(self, data_matrix):
        draw_obj = DrawGraph(data_matrix)
        draw_obj.draw_dots()

    def test_prepare_data(self, data_matrix):
        prepare_obj = PrepareData(data_matrix)
        return prepare_obj.split_data()
    
    def test_model(self, X_test, y_test, X_train, y_train, data_matrix):
        ttm_obj = TrainTestModel(X_test, y_test, X_train, y_train, data_matrix)
        # choose one classification algorithm to classify the data:
        # ttm_obj.train_svm_linear()
        # ttm_obj.train_svm_polynomial()
        # ttm_obj.train_svm_rbf()
        # ttm_obj.train_logistic_regression()
        # ttm_obj.train_knn()
        ttm_obj.train_decision_trees()
        # ttm_obj.train_random_forest()

def main(args):
    test = Test()
    data_matrix = test.test_read_csv(args[1])
    # test.test_draw(data_matrix)
    X_test, y_test, X_train, y_train = test.test_prepare_data(data_matrix)
    test.test_model(X_test, y_test, X_train, y_train, data_matrix)

if __name__ == "__main__":
    if (len(sys.argv) != 2):
         sys.exit(ARGS_NUM_ERROR_MSG)
    main(sys.argv) # [path to this file, arg 1, ...]