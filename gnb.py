# HW#7 Machine Learning 10-601, Meiirbek Islamov
# Gaussian Naive Bayes binary classification algorithm

# import the necessary libraries
import sys
import numpy as np
import csv

args = sys.argv
assert(len(args) == 7)
train_input = args[1] # Path to the training input .csv file
test_input = args[2] # Path to the test input .csv file
train_out = args[3] # Path to output .labels file to which the prediction on the train data should be written
test_out = args[4] # Path to output .labels file to which the prediction on the test data should be written
metrics_out = args[5] # Path of the output .txt file to which metrics such as train and validation error should be written
num_voxels = int(args[6]) # An integer denoting that the top num voxels found via the feature selection method described should be used for training the gnb classifer

# Functions
# Read input csv file
def read(input):
    with open(input, newline='') as f_in:
        read_csv = csv.reader(f_in)
        next(read_csv)
        data = np.array(list(read_csv))
#     data_float = data.astype(float)
    return data[:, :-1].astype(float), data[:, -1]

def prior(labels):
    vals, counts = np.unique(labels, return_counts=True)
    prior_yes = counts[0]/len(labels)
    prior_no = counts[1]/len(labels)
    priors = {}
    priors[vals[0]] = prior_yes
    priors[vals[1]] = prior_no
    return priors

def get_params(feature, label):
    vals, counts = np.unique(label, return_counts=True)
    index_yes, index_no = [], []
    for i, item in enumerate(label):
        if item == vals[0]:
            index_yes.append(i)
        else:
            index_no.append(i)
    feature_yes = np.array([feature[i] for i in index_yes])
    feature_no = np.array([feature[i] for i in index_no])
    mean_yes, mean_no = np.mean(feature_yes), np.mean(feature_no)
    var_yes, var_no = np.var(feature_yes), np.var(feature_no)
    mean, var = {}, {}
    mean[vals[0]] = mean_yes
    mean[vals[1]] = mean_no
    var[vals[0]] = var_yes
    var[vals[1]] = var_no

    return mean, var

def probability(value, mean, var):
    params = {}
    for key, val in mean.items():
        p = (1/np.sqrt(2 * np.pi * var[key])) * np.exp(-((value - val)**2)/(2 * var[key]))
        params[key] = p
    return params


def predict_all(examples, priors, labels, indices, mean_list, var_list):
    predicted_values = []
    for example in examples:
        predicted_values.append(predict(example, priors, labels, indices, mean_list, var_list))
    return predicted_values

def write_labels(predicted_label, filename):
    with open(filename, 'w') as f_out:
        for label in predicted_label:
            f_out.write(str(label) + '\n')

def calculate_error(label_true, label_predicted):
    n = 0
    for i, item in enumerate(label_true):
        if item != label_predicted[i]:
            n += 1
    error = n/len(label_true)
    return error

def write_error(train_error, test_error, filename):
    with open(filename, 'w') as f_out:
        f_out.write("error(train): " + str(train_error) + "\n")
        f_out.write("error(test): " + str(test_error) + "\n")

def k_voxels(k, features, labels):
    mean_diff = []
    for i, item in enumerate(features.T):
        mean, _ = get_params(item, labels)
        values = []
        for key, vals in mean.items():
            values.append(vals)
        diff = abs(values[0] - values[1])
        mean_diff.append(diff)
    sort_index = np.argsort(-np.array(mean_diff))

    only_k_index = []
    for i in range(k):
        only_k_index.append(sort_index[i])

    return only_k_index

def predict(example, priors, labels, indices, mean_list, var_list):
    prob_labels = {}
    sum_yes, sum_no = 0, 0
    priors_list = list(priors)
    for i in indices:
        mean, var = mean_list[i], var_list[i]
        params = probability(example[i], mean, var)
        params_list = list(params)
        sum_yes += np.log(params[params_list[0]])
        sum_no += np.log(params[params_list[1]])

    total_yes = sum_yes + priors[priors_list[0]]
    total_no = sum_no + priors[priors_list[1]]
    prob_labels[priors_list[0]] = total_yes
    prob_labels[priors_list[1]] = total_no
    max_key = max(prob_labels, key=prob_labels.get)

    return max_key

def get_mean_vars(features, labels):
    mean_list, var_list = [], []
    for i, item in enumerate(features.T):
        mean, var = get_params(item, labels)
        mean_list.append(mean)
        var_list.append(var)
    return mean_list, var_list

# Main body
features_train, labels_train = read(train_input)
features_test, labels_test = read(test_input)
priors = prior(labels_train)
indices = k_voxels(num_voxels, features_train, labels_train)
mean_list, var_list = get_mean_vars(features_train, labels_train)
predicted_labels_train = predict_all(features_train, priors, labels_train, indices, mean_list, var_list)
predicted_labels_test = predict_all(features_test, priors, labels_train, indices, mean_list, var_list)
write_labels(predicted_labels_train, train_out)
write_labels(predicted_labels_test, test_out)
train_error = calculate_error(labels_train, predicted_labels_train)
test_error = calculate_error(labels_test, predicted_labels_test)
write_error(train_error, test_error, metrics_out)
