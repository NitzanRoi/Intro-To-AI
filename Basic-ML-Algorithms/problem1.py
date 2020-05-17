# Perceptron Learning Algorithm
# The algorithm learns the linear separator according to the given data (input1.csv) and 
# use this information to classify new points.

import sys
import pandas as pd
import numpy as np
from os import path
from matplotlib import pyplot as plt

CSV_EXIST_ERROR_MSG = "The given path is not a csv file"
ARGS_NUM_ERROR_MSG = "The number of arguments is not valid. Usage: python3 problem1.py <CSV file>"
WEIGHTS_NOT_VALID = "The calculated weights are not valid"
POS_VAL = 1
NEG_VAL = -1

class DrawGraph:
    # Drawing the dots with their linear separator, assuming 2D space
    def __init__(self, feature_matrix, labels_matrix, bias, weights):
        self.feature_matrix = feature_matrix
        self.labels_matrix = labels_matrix
        self.bias = bias
        self.weights = weights

    def split_pos_neg(self):
        pos_idxs = np.asarray(np.where(self.labels_matrix == POS_VAL))
        neg_idxs = np.asarray(np.where(self.labels_matrix == NEG_VAL))
        pos_vector = self.feature_matrix[pos_idxs[0]]
        neg_vector = self.feature_matrix[neg_idxs[0]]
        return pos_vector, neg_vector

    def draw_dots(self): # the given data
        pos_vector, neg_vector = self.split_pos_neg()
        plt.scatter(pos_vector[:, 0], pos_vector[:, 1], color='blue', label='positive')
        plt.scatter(neg_vector[:, 0], neg_vector[:, 1], color='green', label='negative')
        plt.legend()

    def line_equation(self, x, slope, y_intercept):
        return slope * x + y_intercept

    def draw_line(self): # the separator
        if (self.weights[0, 0] == 0 or self.weights[1, 0] == 0):
            sys.exit(WEIGHTS_NOT_VALID)
        slope = -1 * (self.weights[0, 0] / self.weights[1, 0])
        y_intercept = (-1 * self.bias) / self.weights[1, 0]
        x_intercept = (-1 * self.bias) / self.weights[0, 0]

        min_val_coords = np.where(self.feature_matrix == np.min(self.feature_matrix[:, 0]))
        x_min = self.feature_matrix[list(zip(min_val_coords[0], min_val_coords[1]))[0]]

        max_val_coords = np.where(self.feature_matrix == np.max(self.feature_matrix[:, 0]))
        x_max = self.feature_matrix[list(zip(max_val_coords[0], max_val_coords[1]))[0]]
        x_max = np.max([x_max, 0, x_intercept])

        distance = 2
        x_coord = np.min([x_min, 0, x_intercept])
        y_coord = self.line_equation(x_coord, slope, y_intercept)
        x_arr = [x_coord]
        y_arr = [y_coord]
        for i in range(int((x_max - x_min) / distance) + 1):
            x_coord += distance
            y_coord = self.line_equation(x_coord, slope, y_intercept)
            x_arr.append(x_coord)
            y_arr.append(y_coord)
        plt.plot(np.array(x_arr), np.array(y_arr), color='red')

class Perceptron:
    # representing the perceptron
    def __init__(self, feature_matrix, labels_matrix):
        self.feature_matrix = feature_matrix
        self.labels_matrix = labels_matrix
        self.bias = 0
        self.weights = np.zeros((feature_matrix.shape[1], 1))
    
    def calc_weights(self):
        is_converge = False
        while (not is_converge):
            is_converge = True
            for i in range(self.feature_matrix.shape[0]):
                multiply_sum = np.matmul(self.feature_matrix[i, :], self.weights) + self.bias
                if (self.is_error(self.labels_matrix[i, 0], multiply_sum)):
                    self.update_weights(self.labels_matrix[i, 0], self.feature_matrix[i, :])
                    is_converge = False
        return self.bias, self.weights
    
    def is_error(self, true_label, multiply_sum):
        tmp_label = 1 if multiply_sum > 0 else -1 # step function
        return (true_label * tmp_label <= 0)
    
    def update_weights(self, true_label, x_i):
        self.weights += true_label * x_i.reshape((x_i.shape[0], 1))
        self.bias += true_label

    def evaluate_new_point(self, new_point):
        slope = -1 * (self.weights[0, 0] / self.weights[1, 0])
        y_intercept = (-1 * self.bias) / self.weights[1, 0]
        label = - 1 if (slope * new_point[0] + y_intercept) > new_point[1] else 1
        print('new-point label: ', str(label))

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
        csv_file = pd.read_csv(self.csv_path, header=None)
        for i in range(csv_file.shape[1] - 1):
            csv_file.rename(columns={csv_file.columns[i]: "feature_"+str(i)}, 
                            inplace=True)
        csv_file.rename(columns={csv_file.columns[csv_file.shape[1] - 1]: "labels"}, 
                        inplace=True)
        return csv_file

    def data_to_matrices(self, csv_file):
        features_matrix = np.zeros((csv_file.shape[0], csv_file.shape[1] - 1))
        labels_vector = csv_file['labels'].values
        for i in range(csv_file.shape[1] - 1):
            features_matrix[:, i] = csv_file["feature_"+str(i)].values
        return features_matrix, labels_vector.reshape((labels_vector.shape[0], 1))

class Test:
    # class for tests
    def __init__(self):
        pass

    def test_read_csv(self, file_name):
        csv_obj = ParseCSV(file_name)
        csv_file = csv_obj.read_csv()
        return csv_obj.data_to_matrices(csv_file)

    def test_perceptron(self, features_matrix, labels_vector):
        perceptron = Perceptron(features_matrix, labels_vector)
        bias, weights = perceptron.calc_weights()
        new_point = (10, 10)
        perceptron.evaluate_new_point(new_point)

        draw = DrawGraph(features_matrix, labels_vector, bias, weights)
        draw.draw_dots()
        draw.draw_line()

    def test_print_csv(self, csv_file):
        print(csv_file)
        print(csv_file.columns[1])
        print(csv_file['labels'].values) # numpy vector
        print(csv_file['label']) # pandas series

#### main ####

def main(args):
    test = Test()
    features_matrix, labels_vector = test.test_read_csv(args[1])
    test.test_perceptron(features_matrix, labels_vector)
    # test.test_print_csv(csv_file)

if __name__ == "__main__":
    if (len(sys.argv) != 2):
         sys.exit(ARGS_NUM_ERROR_MSG)
    main(sys.argv) # [path to this file, arg 1, ...]