import sys
import pandas as pd
import numpy as np
from os import path
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

CSV_EXIST_ERROR_MSG = "The given path is not a csv file"
ARGS_NUM_ERROR_MSG = "The number of arguments is not valid. Usage: python3 problem2.py <CSV file>"

class DrawGraph:
    # class for drawing the model
    # features_matrix is the original feature-matrix, before the scaling
    # assuming the weights vector has 3 dimensions
    def __init__(self, features_matrix, labels_vector, weights):
        self.features_matrix = features_matrix
        self.labels_vector = labels_vector.T[0]
        self.weights = weights[2] # todo try for all weights
        self.fig = plt.figure(figsize=(8,8))

    def draw_graph(self):
        ax = self.fig.add_subplot(111, projection='3d')
        ax.set_title('Linear Regression')
        ax.set_xlabel('Age (Years)')
        ax.set_ylabel('Weight (Kilograms)')
        ax.set_zlabel('Height (Meters)')
        ax.scatter(self.features_matrix[:, 1], self.features_matrix[:, 2], 
                   self.labels_vector, color='r') # draw dots
        
        a, b, c, d = self.weights[1,0], self.weights[2,0], -1, self.weights[0,0]
        mn_x, mx_x = np.min(self.features_matrix[:, 1]), np.max(self.features_matrix[:, 1])
        mn_y, mx_y = np.min(self.features_matrix[:, 2]), np.max(self.features_matrix[:, 2])
        x, y = np.linspace(mn_x, mx_x, 10), np.linspace(mn_y, mx_y, 10)
        X, Y = np.meshgrid(x, y)
        Z = (d - a*X - b*Y) / c
        ax.plot_surface(X, Y, Z) # draw plane
        plt.show()

class LinearRegression:
    # class for calculating the regression model
    # the given parameter scaled_matrix is a scaled features_matrix
    def __init__(self, scaled_matrix, labels_vector):
        self.scaled_matrix = scaled_matrix
        self.labels_vector = labels_vector

    def calc_cost_function(self, beta_values):
        predictions = np.dot(self.scaled_matrix, beta_values)
        cost = (1 / (2 * self.scaled_matrix.shape[0])) * \
                np.sum(np.square(predictions - self.labels_vector), axis=0)
        return cost

    def gradient_descent(self):
        alpha_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10] # trying some different learning-rates
        iterations = 100
        final_cost = {}
        final_weights = []
        for alpha in alpha_values:
            beta_values = np.zeros((self.scaled_matrix.shape[1], 1))
            for i in range(iterations):
                predictions = np.dot(self.scaled_matrix, beta_values)
                beta_values = beta_values - (alpha / self.scaled_matrix.shape[0]) * \
                              np.dot(self.scaled_matrix.T, (predictions - self.labels_vector))
            final_cost[alpha] = self.calc_cost_function(beta_values)
            final_weights.append(beta_values)
        return final_cost, final_weights

    def predict(self, weights, point): # weights=[w0,w1,w2], point=(x,y)
        return weights[1]*point[0] + weights[2]*point[1] + weights[0]


class Normalization:
    # class for data normalization (feature scaling).
    # The given data has different scales: age (years) and weight (kilograms), 
    # so we have to normalize them to the same scale (mean=0, std-dev=1 of feature's columns)
    def __init__(self, features_matrix):
        self.features_matrix = features_matrix

    def normalize_features(self):
        scaled_matrix = np.ones(self.features_matrix.shape)
        scaled_matrix[:, 1:] = (self.features_matrix[:, 1:] - 
                                np.mean(self.features_matrix[:, 1:], axis=0)) \
                                / np.std(self.features_matrix[:, 1:], axis=0)
        return scaled_matrix

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
        col_names = ['age', 'weight', 'height']
        csv_file = pd.read_csv(self.csv_path, names=col_names, header=None)
        return csv_file

    def data_to_matrices(self, csv_file):
        csv_cols = list(csv_file.columns)
        features_matrix = np.ones(csv_file.shape)
        labels_vector = csv_file['height'].values
        for i in range(csv_file.shape[1] - 1):
            features_matrix[:, i+1] = csv_file[csv_cols[i]].values
        return features_matrix, labels_vector.reshape((labels_vector.shape[0], 1))

class Test:
    # class for tests
    def __init__(self):
        self.lr = None

    def test_print(self, text, data):
        print(text)
        print(data)
        print('----')

    def test_read_csv(self, csv_path):
        csv_obj = ParseCSV(csv_path)
        csv_file = csv_obj.read_csv()
        return csv_obj.data_to_matrices(csv_file)

    def test_normalization(self, features_matrix=None):
        normalization = Normalization(features_matrix)
        scaled_matrix = normalization.normalize_features()
        return scaled_matrix

    def test_linear_regression(self, scaled_matrix, labels_vector):
        self.lr = LinearRegression(scaled_matrix, labels_vector)
        return self.lr.gradient_descent()

    def test_draw(self, features_matrix, labels_vector, weights):
        dg = DrawGraph(features_matrix, labels_vector, weights)
        dg.draw_graph()

    def test_predict(self, weights, point):
        prediction = self.lr.predict(weights, point) # prediction for a new point
        pt = np.array([[point[0], point[1], prediction]]) # a point on the plane
        normm = np.array([[weights[1]],[weights[2]],[-1]]) # the normal to the plane
        d = weights[0] # the intercept
        self.test_print("point * normal (should be around 0)", np.dot(pt, normm) + d)
        return prediction


def main(args):
    test = Test()
    features_matrix, labels_vector = test.test_read_csv(args[1])
    scaled_matrix = test.test_normalization(features_matrix)
    final_cost, final_weights = test.test_linear_regression(scaled_matrix, labels_vector)
    test.test_draw(features_matrix, labels_vector, final_weights)
    
    test.test_print("cost", final_cost)
    new_point = (2.13, 11.44697)
    pdt = test.test_predict(final_weights[2].T[0], new_point)
    test.test_print("prediction", pdt)

if __name__ == "__main__":
    if (len(sys.argv) != 2):
         sys.exit(ARGS_NUM_ERROR_MSG)
    main(sys.argv) # [path to this file, arg 1, ...]