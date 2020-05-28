#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.neighbors import KernelDensity

VERSION = '1.0.0'

print('\ncytoShapeNet v.' + VERSION)
print('© 2018-2020 Dr. Stephan Quint & Konrad Hinkelmann')
print('info@cytoshape.net\n\n')

# make use of longer loading imports for displaying the info text
import seaborn as sns
from keras.models import model_from_json


__doc__ = 'Evaluation part of cytoShapeNet' \
          '===============================' \
          '' \
          'This script uses the neural networks created in the previous training part to predict and plot cell data ' \
          'from "PATH".'
__author__ = 'Stephan Quint, Konrad Hinkelmann'
__copyright__ = 'Copyright © 2018-2020, cytoShapeNet'
__credits__ = ['Stephan Quint, Konrad Hinkelmann, Greta Simionato, Revaz Chachanidze, Paola Bianchi, Elisa Fermo,'
               'Richard van Wijk, Christian Wagner, Lars Kaestner, Marc Leonetti']
__license__ = 'GPL'
__version__ = VERSION
__email__ = 'info@cytoshape.net'


# constants
ALT_LABELS = True
ALT_LABEL_VALUES = ['SDE', 'CC', 'KE', 'KN', 'ML', 'AC', 'U']
CONFUSION_THRESHOLD = 0.75

# this script will be called from the root directory .bat file, where the "PATH", "NN_PATH" folders are located
DATA_PATH = 'data\\evaluation\\'
NN_PATH = 'nn\\'
FONT = {'fontname': 'DejaVu Sans', 'fontsize': '9'}
FONT_SMALL = {'fontname': 'DejaVu Sans', 'fontsize': '7'}
FONT_SMALL_ITALIC = {'fontname': 'DejaVu Sans', 'fontsize': '7', 'fontstyle': 'italic'}
PLOT_COLORS = ['darkblue', 'darkred', 'darkgreen', 'darkorange']


# forcing Keras / Tensorflow to run on CPU to get reproducible results
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

sns.set_style('darkgrid', {'axes.facecolor': '.9'})
plot_color_idx = 0
processed_cell_count = 0


def file_to_list(filename):
    file_handle = open(filename, 'r')
    out_list = []
    for line in file_handle.readlines():
        out_list.append(line.replace('\n', ''))
    file_handle.close()
    return out_list


def mm2inch(value):
    return value / 25.4


def find_nearest(array, value):
    array = np.asarray(array)
    nearest_idx = (np.abs(array - value)).argmin()
    return array[nearest_idx]


def convert_digits(text):
    if text.isdigit():
        text = int(text)
    return text


def sort_alphanumerically(input_list):
    alphanum_key = lambda key: [convert_digits(c) for c in re.split('([0-9]+)', key)]
    return sorted(input_list, key=alphanum_key)



for directory in os.listdir(DATA_PATH):
    data_directory = DATA_PATH + directory + '\\'
    if os.path.isdir(data_directory):
        # compute all folders (separately) in the "DATA_PATH" directory
        data_collection = {}
        # different plot color for each folder
        plot_color = PLOT_COLORS[plot_color_idx % len(PLOT_COLORS)]
        # load all data from data_directory
        for root, directories, files in os.walk(data_directory):
            for file in files:
                if file.endswith('.dat'):
                    # remove 'data\'
                    input_folder = os.path.basename(root)
                    # load data
                    print('Loading: ' + os.path.join(root, file))
                    data = np.loadtxt(os.path.join(root, file), skiprows=1)
                    # normalize data
                    data /= np.max(abs(data))
                    if not (input_folder in data_collection):
                        data_collection[input_folder] = []
                    data_collection[input_folder].append(data)

        # load classification NN
        json_file = open(NN_PATH + 'classification_NN.json', 'r')
        classification_model = model_from_json(json_file.read())
        json_file.close()
        classification_model.load_weights(NN_PATH + 'classification_NN.h5')

        # load regression NN
        json_file = open(NN_PATH + 'regression_NN.json', 'r')
        regression_model = model_from_json(json_file.read())
        json_file.close()
        regression_model.load_weights(NN_PATH + 'regression_NN.h5')

        labels = file_to_list(NN_PATH + 'class_labels.txt')
        all_SDE_collections = []

        SDE_densities_df = pd.DataFrame()
        SDE_histograms_df = pd.DataFrame()
        classes_histograms_df = pd.DataFrame()

        # sort according to windows
        data_collection_keys = sort_alphanumerically(data_collection.keys())

        SDE_cell_count_collection = []
        for subject in data_collection_keys:
            print('\n\nPredicting cells of ' + subject, end='', flush=True)
            histogram = {key: 0 for key in labels}
            SDE_data_collection = []
            unknown = 0
            for idx in range(len(data_collection[subject])):
                print('.', end='', flush=True)
                data = np.asarray([data_collection[subject][idx]])
                prediction = classification_model.predict(data)
                processed_cell_count += 1
                if np.max(prediction) < CONFUSION_THRESHOLD:
                    unknown += 1
                    continue

                result = [np.argmax(y) for y in prediction][0]
                histogram[labels[result]] += 1

                if labels[result] == 'SDE shapes':
                    SDE_data_collection.append(data)

            # normalize classes histogram
            histogram['unknown'] = unknown
            classes_hist_values_norm = list(histogram.values()) / np.sum(list(histogram.values()))

            # SDE distribution
            SDE_results = []
            for idx in range(len(SDE_data_collection)):
                print('.', end='', flush=True)
                SDE_prediction = regression_model.predict(SDE_data_collection[idx])
                SDE_results.append(SDE_prediction)

            subject = subject.replace('_', ' ')

            print('\n\nAnalyzing cells of ' + subject, flush=True)
            # probability density estimation
            x_grid = np.linspace(-1.25, 1.25, 1251)[:, np.newaxis]
            kde = KernelDensity(bandwidth=0.025, kernel='gaussian')
            SDE_results = np.asarray(np.squeeze(SDE_results)).reshape(-1, 1)
            kde.fit(SDE_results)
            log_prob = kde.score_samples(x_grid)
            kde_dist = np.exp(log_prob)

            all_SDE_collections.append(np.exp(log_prob))
            ppath = subject.replace('\\', '_')
            dest_path_SDE_raw = data_directory + 'SDE_raw_' + ppath + '.csv'
            np.savetxt(dest_path_SDE_raw, SDE_results)

            # identifier = folder prefix (C1, C2, ...)
            identifier = subject.split(' ')[0]
            SDE_densities_df[identifier] = kde_dist
            SDE_hist, bins = np.histogram(SDE_results, bins=np.linspace(-1.25, 1.25, 126), density=True)
            SDE_hist = np.append(SDE_hist, [0])
            SDE_histograms_df[identifier] = SDE_hist
            classes_histograms_df[identifier] = classes_hist_values_norm

            SDE_cell_count_collection.append(len(SDE_results))
        
        print('\nExporting CSV files for "' + directory + '"')
        # append independent variables to dataframes
        SDE_densities_df['x'] = x_grid
        SDE_histograms_df['bins'] = bins
        classes_histograms_df['bins'] = histogram.keys()

        # save_dataframes to csv files
        dest_path_SDE_densities = data_directory + 'SDE_densities.csv'
        dest_path_SDE_histograms = data_directory + 'SDE_histograms.csv'
        dest_path_classes_histograms = data_directory + 'classes_histograms.csv'

        SDE_densities_df.to_csv(dest_path_SDE_densities, index=False)
        SDE_histograms_df.to_csv(dest_path_SDE_histograms, index=False)
        classes_histograms_df.to_csv(dest_path_classes_histograms, index=False)

        print('Generating line plot for "' + directory + '"')
        subject_names = SDE_densities_df.head(0).columns
        # minimum of "1" for reasonable visualisation of the plot height when using a single input folder
        subject_count = len(subject_names) - 1
        height_count = max(subject_count, 2)

        height = mm2inch(22 * height_count + 5)
        fig, axes = plt.subplots(1, 2, figsize=(mm2inch(180), height))

        subscale = 7
        y_offset = subscale * (subject_count-1)
        ax1 = plt.subplot(1, 2, 2)

        x_grid = np.linspace(-1.25, 1.25, 1251, dtype=np.single)

        for idx in range(subject_count):
            bardata = classes_histograms_df[subject_names[idx]]
            plt.plot(x_grid, SDE_densities_df[subject_names[idx]] + y_offset, color=plot_color, linewidth=0.5)
            plt.fill_between(x_grid, y_offset, SDE_densities_df[subject_names[idx]] + y_offset, alpha=0.2,
                             color=plot_color)
            subject_id = subject_names[idx]
            subject_id = subject_id.split(' ')[0]

            SDE_cell_count = SDE_cell_count_collection[idx]

            plt.text(0.65, y_offset + 0.65*subscale, 'total: %d' % SDE_cell_count, **FONT_SMALL)

            # calculate expect value
            density = SDE_densities_df[subject_names[idx]]
            res = x_grid[1]-x_grid[0]
            expect_val = np.trapz(np.multiply(density, x_grid), dx=res)
            std_dev = np.sqrt(np.trapz(np.multiply(np.power(np.subtract(x_grid, expect_val), 2), density), dx=res))

            for jdx in range(len(density)):
                probability = np.trapz(density[0:jdx], x_grid[0:jdx])
                if probability >= 0.025:
                    prob_0025_x_val = x_grid[jdx]
                    jdx_save = jdx
                    break

            for jdx in range(len(density)):
                probability = np.trapz(density[jdx_save:jdx+jdx_save], x_grid[jdx_save:jdx+jdx_save])
                if probability >= 0.95:
                    prob_0975_x_val = x_grid[jdx+jdx_save]
                    break

            lb = np.where(x_grid == find_nearest(x_grid, prob_0025_x_val))[0][0]
            ub = np.where(x_grid == find_nearest(x_grid, prob_0975_x_val))[0][0]
            inner_integral = np.trapz(density[lb:ub], x_grid[lb:ub], dx=res)
            ax1.plot([expect_val, expect_val], [y_offset, y_offset + 0.9 * subscale], '--', linewidth=0.5,
                     color=plot_color)
            ax1.add_patch(patches.Rectangle((x_grid[lb], y_offset), (x_grid[ub]-x_grid[lb]),  0.9 * subscale,
                                            facecolor='black', fill=True, alpha=0.1, linewidth=0.0))
            plt.text(-1.2, y_offset + 0.65 * subscale, '$\mu$ = %.2f' % expect_val, **FONT_SMALL)
            plt.text(-1.2, y_offset + 0.45 * subscale, 'CI$_{0.95}$ = [%.2f,%.2f]' % (x_grid[lb], x_grid[ub]),
                     **FONT_SMALL)

            mutation_label_filepath = data_directory + 'mutation_labels.txt'
            if os.path.exists(mutation_label_filepath):
                mutation_labels = []
                mutation_label_file = open(mutation_label_filepath)
                items = mutation_label_file.readlines()
                for item in items:
                    mutation_labels.append(item.split('\t')[1].strip())
                plt.text(-1.2, y_offset + 0.25 * subscale, mutation_labels[idx], **FONT_SMALL_ITALIC)
            y_offset = y_offset - subscale

        plt.xlim(np.min(x_grid), np.max(x_grid))
        plt.ylim(-subscale*0.1, subscale * subject_count + subscale / 4)
        plt.yticks(np.linspace(0, subscale*(subject_count-1), subject_count), [])
        SDE_label_list = []
        SDE_labels = [-1.00, -0.67, -0.33, 0.00, 0.33, 0.67, 1.00]
        SDE_labels_custom = ['SP', 'ST II', 'ST I', 'D', 'E I', 'E II', 'E III']
        for idx in range(len(SDE_labels_custom)):
            SDE_label_list.append('%.2f\n' % (SDE_labels[idx]) + SDE_labels_custom[idx])
        plt.xticks(SDE_labels, **FONT_SMALL)
        ax = plt.gca()
        ax.set_xticklabels(SDE_label_list,  **FONT_SMALL)

        plt.yticks(np.linspace(0, subscale*(subject_count-1), subject_count), [])
        plt.ylabel('probability density', **FONT)
        plt.xlabel('SDE distribution', **FONT)
        ax1.xaxis.set_label_position('top')
        plt.grid(True)

        ax2 = plt.subplot(1, 2, 1)
        y_offset = subscale * (subject_count-1)

        print('Generating bar plot for "' + directory + '"')
        for idx in range(subject_count):
            bardata = classes_histograms_df[subject_names[idx]]
            for jdx in range(len(bardata)):
                if jdx == 0:
                    color = plot_color
                else:
                    color = 'black'
                ax2.add_patch(patches.Rectangle((jdx+0.75, y_offset), 0.5, bardata[jdx] * subscale * 0.9,
                                                edgecolor=color, facecolor='white', linewidth=0.5))
                ax2.add_patch(patches.Rectangle((jdx+0.75, y_offset), 0.5, bardata[jdx] * subscale * 0.9,
                                                edgecolor=color, facecolor=color, fill=True, alpha=0.2, linewidth=0.5))
                plt.text(jdx + 0.75 + 0.14, y_offset + 1, '%.3f' % bardata[jdx], **FONT_SMALL, rotation=90)
            subject_id = subject_names[idx]
            subject_id = subject_id.split(' ')[0]
            total_cell_count = str(len(data_collection[data_collection_keys[idx]]))
            plt.text(0.1 + 6, y_offset + 0.65 * subscale, 'total: ' + total_cell_count, **FONT_SMALL)
            plt.text(0.1, y_offset + 0.5, subject_id, **FONT_SMALL)
            y_offset = y_offset - subscale

        plt.xlim(0, len(bardata) + 1)
        if ALT_LABELS:
            plt.xticks([1, 2, 3, 4, 5, 6, 7], ALT_LABEL_VALUES, **FONT_SMALL)
        else:
            plt.xticks([1, 2, 3, 4, 5, 6, 7], histogram.keys(), rotation=90, **FONT_SMALL)
        plt.ylim(-subscale*0.1, subscale * subject_count + subscale / 4)
        plt.yticks(np.linspace(0, subscale * (subject_count-1), subject_count), [])
        plt.yticks(np.linspace(0, subscale * (subject_count-1), subject_count), [])
        plt.ylabel('probability', **FONT)
        ax2.xaxis.set_label_position('top')
        plt.xlabel('classification', **FONT)

        plt.tight_layout()
        plt.savefig(data_directory + '\\evaluation_results.pdf')
        plot_color_idx += 1

print('\nProcessed ' + str(processed_cell_count) + ' cells')
plt.show()
