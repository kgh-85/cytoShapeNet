##!/usr/bin/python
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras import utils
from keras.optimizers import Adam
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks


VERSION = '1.0.0'

print('\ncytoShapeNet v.' + VERSION)
print('© 2018-2020 Dr. Stephan Quint & Konrad Hinkelmann')
print('info@cytoshape.net\n\n')

# make use of longer loading imports for displaying the info text
import seaborn as sns
from keras.models import model_from_json


__doc__ = 'Training part of cytoShapeNet' \
          '===============================' \
          '' \
          'This script trains the neural networks.'
__author__ = 'Stephan Quint, Konrad Hinkelmann'
__copyright__ = 'Copyright © 2018-2020, cytoShapeNet'
__credits__ = ['Stephan Quint, Konrad Hinkelmann, Greta Simionato, Revaz Chachanidze, Paola Bianchi, Elisa Fermo,'
               'Richard van Wijk, Christian Wagner, Lars Kaestner, Marc Leonetti']
__license__ = 'GPL'
__version__ = VERSION
__email__ = 'info@cytoshape.net'


# this script will be called from the root directory 01_training.bat file, where the "DATA_PATH" folder is located
DATA_PATH = 'data\\training\\'
BENCHMARK_PATH = 'data\\benchmark_regression\\'
FONT = {'fontname': 'DejaVu Sans'}
FONT_BOLD = {'fontname': 'DejaVu Sans', 'weight': 'bold'}
REPRODUCIBLE = True
ADD_NOISE = False
NOISE_STD = 0.1

# augmentation
INTERPOLATIONS = 1000
CONFUSION_THRESHOLD = 0.75

MIN_PEAK_WIDTH = 40
BATCH_SIZE = 100

EPOCHS_CLASSES = 100
EPOCHS_SDE = 40

# validation split
VALIDATION_RATIO = 0.2
VALIDATION_AMOUNT = INTERPOLATIONS * VALIDATION_RATIO


def validation_amount(value):
    # exclusive validation data
    global VALIDATION_RATIO
    return round(VALIDATION_RATIO * value)


def cm_to_inch(value):
    return value / 2.54


def shuffle(x, y, random_seed):
    z = list(zip(x, y))
    random.Random(random_seed).shuffle(z)
    x, y = zip(*z)
    x = np.asarray(x)
    y = np.asarray(y)
    return x, y


# init benchmark values
min_validation_loss_classes = np.inf
min_validation_loss_SDE = np.inf

if REPRODUCIBLE:
    # forcing Keras / Tensorflow to run on CPU to get reproducible results
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['PYTHONHASHSEED'] = '0'

    CUSTOM_RANDOM_SEED = 136456
    THREAD_COUNT = 1
    np.random.seed(CUSTOM_RANDOM_SEED)
    random.seed(CUSTOM_RANDOM_SEED)
    tf.compat.v1.set_random_seed(CUSTOM_RANDOM_SEED)
else:
    THREAD_COUNT = os.cpu_count()
    CUSTOM_RANDOM_SEED = None


# define collections for SDE training and validation
training_x_SDE = []
training_y_SDE = []
validation_x_SDE = []
validation_y_SDE = []
SDE_classes = {}
SDE_full_labels = {}

# define collections for classes training and validation
training_x_classes = []
training_y_classes = []
validation_x_classes = []
validation_y_classes = []
weight_classes = []
cell_A = []
cell_B = []
source_files_classes = {}
classes = {}

# Get all input files
i = 0
for root, directories, files in os.walk(DATA_PATH):
    for file in files:
        if file.endswith('.dat'):
            input_folder = os.path.basename(root)
            # get possible float part (separated by "_")
            label = input_folder.split('_', 1)[0]
            # load data
            print('Loading: ' + os.path.join(root, file))
            data = np.loadtxt(os.path.join(root, file), skiprows=1)
            # normalize data
            data /= np.max(abs(data))
            # test if it actually is a float
            try:
                SDE_label = float(label)
                # initialize empty list for dictionary key
                if not (SDE_label in SDE_classes):
                    SDE_classes[SDE_label] = []
                    SDE_full_labels[SDE_label] = input_folder.split('_', 1)[1]
                    source_files_classes[SDE_label] = []
                # append data to label in dictionary
                SDE_classes[SDE_label].append(data)
                source_files_classes[SDE_label].append(file)
            except ValueError:
                if not (input_folder in classes):
                    classes[input_folder] = []
                    source_files_classes[input_folder] = []
                classes[input_folder].append(data)
                source_files_classes[input_folder].append(file)

# sort all labels
SDE_labels = sorted(SDE_classes.keys())
class_labels = sorted(classes.keys())

current_validation_samples = []
next_validation_samples = []

# process all SDE classes

for SDE_label_idx in range(len(SDE_labels) - 1):

    # get data count of current and next group
    current_data_count = len(SDE_classes[SDE_labels[SDE_label_idx]])
    next_data_count = len(SDE_classes[SDE_labels[SDE_label_idx + 1]])

    SDE_label = SDE_labels[SDE_label_idx]
    SDE_label_full = str(SDE_labels[SDE_label_idx]) + '_' + SDE_full_labels[SDE_label]

    next_SDE_label = SDE_labels[SDE_label_idx + 1]
    next_SDE_label_full = str(SDE_labels[SDE_label_idx + 1]) + '_' + SDE_full_labels[next_SDE_label]

    print('[' + str(SDE_label_idx + 1) + '/' + str(len(SDE_labels) - 1) + '] augmenting ' + str(
        INTERPOLATIONS) + ' intermediate training samples out of ' + str(
        current_data_count + next_data_count) + ' input samples for \"' + str(
        SDE_labels[SDE_label_idx]) +
          '\" --> \"' + str(SDE_labels[SDE_label_idx + 1]) + '\" ...')

    # iterate through float labels (except the last one)
    delta = SDE_labels[SDE_label_idx + 1] - SDE_labels[SDE_label_idx]

    # stick with the same validation samples for the next group
    current_validation_samples = next_validation_samples
    next_validation_samples = []

    # get exclusive validation samples for current group
    if len(current_validation_samples) == 0:
        while len(current_validation_samples) < validation_amount(current_data_count):
            idx = random.randint(0, current_data_count - 1)
            if idx in current_validation_samples:
                continue
            current_validation_samples.append(idx)

    # get exclusive validation samples for next group
    while len(next_validation_samples) < validation_amount(next_data_count):
        idx = random.randint(0, next_data_count - 1)
        if idx in next_validation_samples:
            continue
        next_validation_samples.append(idx)

    # create augmented data

    for idx in range(INTERPOLATIONS):
        add_validation_sample = False
        if (len(training_y_classes) > 0 or len(training_y_SDE) > 0) and current_validation_samples:
            len_validation = len(validation_y_SDE)
            len_training = len(training_y_SDE)
            if len_validation / (len_training + len_validation) < VALIDATION_RATIO:
                add_validation_sample = True

        # get random indices
        while True:
            current_data_idx = random.randint(0, current_data_count - 1)
            if (current_data_idx in current_validation_samples) == add_validation_sample:
                break

        while True:
            next_data_idx = random.randint(0, next_data_count - 1)
            if (next_data_idx in next_validation_samples) == add_validation_sample:
                break

        current_data = SDE_classes[SDE_labels[SDE_label_idx]][current_data_idx]
        next_data = SDE_classes[SDE_labels[SDE_label_idx + 1]][next_data_idx]

        # create augmented data
        rand_weight = random.uniform(0, 1)
        weighted_data = current_data + (next_data - current_data) * rand_weight

        if add_validation_sample:
            validation_x_SDE.append(weighted_data)
            # add the corresponding label
            validation_y_SDE.append(SDE_labels[SDE_label_idx] + (delta * rand_weight))
        else:
            if ADD_NOISE:
                # add noise for training
                weighted_data = weighted_data * (1 + np.random.normal(0, NOISE_STD, len(weighted_data)))
            training_x_SDE.append(weighted_data)
            # add the corresponding label
            training_y_SDE.append(SDE_labels[SDE_label_idx] + (delta * rand_weight))

        if add_validation_sample:

            # check to add another validation sample for classification network
            if len(validation_y_classes) / (len(training_y_classes) / (1 - VALIDATION_RATIO)) < VALIDATION_RATIO:
                validation_x_classes.append(weighted_data)
                validation_y_classes.append(int(0))

                # save weight and source cells for csv export
                weight_classes.append(rand_weight)
                cell_A.append(SDE_label_full + '\\' + source_files_classes[SDE_label][current_data_idx])
                cell_B.append(next_SDE_label_full + '\\' + source_files_classes[next_SDE_label][next_data_idx])
        else:
            # check to add another train sample for the classification network
            if len(training_y_classes) / len(training_y_SDE) < 1 / (len(SDE_labels) - 1):
                if ADD_NOISE:
                    weighted_data = weighted_data * (1 + np.random.normal(0, NOISE_STD, len(weighted_data)))
                training_x_classes.append(weighted_data)
                training_y_classes.append(int(0))

# process all classes

for class_label_idx in range(len(class_labels)):
    # get data count
    data_count = len(classes[class_labels[class_label_idx]])
    if data_count < 2:
        print('Warning: Skipping folder \"' + class_labels[
            class_label_idx] + '\" (less than two samples inside)')
        continue

    validation_samples = []
    # reserve exclusive samples for validation if prerequisites are met
    while len(validation_samples) < validation_amount(data_count):
        cell_idx = random.randint(0, data_count - 1)
        if cell_idx in validation_samples:
            continue
        validation_samples.append(cell_idx)

    print('[' + str(class_label_idx + 1) + '/' + str(len(class_labels)) + '] Augmenting ' + str(
        INTERPOLATIONS) + ' training samples out of ' + str(data_count) + ' input samples for \"' +
          class_labels[class_label_idx] + '\"...')

    class_label = class_labels[class_label_idx]

    for idx in range(INTERPOLATIONS):
        add_validation_sample = False
        if training_y_classes and validation_samples:
            len_validation_y_classes = len(validation_y_classes)
            len_training_y_classes = len(training_y_classes)

            if training_y_SDE:
                len_validation_y_classes -= VALIDATION_AMOUNT
                len_training_y_classes -= INTERPOLATIONS - VALIDATION_AMOUNT

            if len_training_y_classes and len_validation_y_classes / (
                    len_training_y_classes + len_validation_y_classes) < VALIDATION_RATIO:
                add_validation_sample = True

        # find two different pairs in same class
        while True:
            current_data_idx = random.randint(0, data_count - 1)
            if (current_data_idx in validation_samples) != add_validation_sample:
                continue

            next_data_idx = random.randint(0, data_count - 1)
            if (next_data_idx in validation_samples) != add_validation_sample:
                continue

            if current_data_idx != next_data_idx:
                break

        current_data = classes[class_label][current_data_idx]
        next_data = classes[class_label][next_data_idx]

        # create augmented data
        rand_weight = random.uniform(0, 1)
        weighted_data = current_data + (next_data - current_data) * rand_weight

        if add_validation_sample:
            validation_x_classes.append(weighted_data)
            validation_y_classes.append(int(class_label_idx + 1))

            # save weight and source cells for later debug csv export
            weight_classes.append(rand_weight)
            cell_A.append(class_label + '\\' + source_files_classes[class_label][current_data_idx])
            cell_B.append(class_label + '\\' + source_files_classes[class_label][next_data_idx])
        else:
            if ADD_NOISE:
                weighted_data = weighted_data * (1 + np.random.normal(0, NOISE_STD, len(weighted_data)))
            training_x_classes.append(weighted_data)
            training_y_classes.append(int(class_label_idx + 1))

print('Preparing train data...')

# shuffle with specific random seed to get reproducible results and convert to numpy array
np_training_x_SDE, np_training_y_SDE = shuffle(training_x_SDE, training_y_SDE, CUSTOM_RANDOM_SEED)
np_validation_x_SDE = np.asarray(validation_x_SDE)
np_validation_y_SDE = np.asarray(validation_y_SDE)

np_training_x_classes, np_training_y_classes = shuffle(training_x_classes, training_y_classes, CUSTOM_RANDOM_SEED)
cat_training_y_classes = utils.to_categorical(np_training_y_classes)

np_validation_x_classes = np.asarray(validation_x_classes)
np_validation_y_classes = np.asarray(validation_y_classes)
cat_validation_y_classes = utils.to_categorical(np_validation_y_classes)

# general approach: wait for reaching patience
early_stopping_monitor = EarlyStopping(patience=5)

input_value_count = len(np_training_x_classes[0])
classes_label_count = len(class_labels) + 1

# define, compile and train the classification model
classification_model = Sequential()
classification_model.add(Dense(int(input_value_count/10), input_dim=input_value_count, use_bias=True,
                         activation='relu'))

classification_model.add(Dense(classes_label_count, use_bias=True, kernel_initializer='normal', activation='softmax'))

opt = Adam()
classification_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['acc'])

print('Training classification network on ' + str(len(cat_training_y_classes)) + ' samples')
classification_history = classification_model.fit(np_training_x_classes, cat_training_y_classes, epochs=EPOCHS_CLASSES,
                                                  validation_split=0,
                                                  validation_data=(np_validation_x_classes, cat_validation_y_classes),
                                                  batch_size=BATCH_SIZE, verbose=1,
                                                  shuffle=True)

# save the sort out model
with open('classification_NN.json', 'w') as json_file:
    json_file.write(classification_model.to_json())

# serialize weights to HDF5
classification_model.save_weights('classification_NN.h5')

# remove "A_" , "B_", ... from training folder names
class_labels = [label[2:].replace('_', ' ') for label in class_labels]
class_labels.insert(0, 'SDE shapes')

# save labels
label_captions = open('class_labels.txt', 'w')
for label in class_labels:
    label_captions.write(label + '\n')
label_captions.close()

# define, compile and train the regression model
regression_model = Sequential()
regression_model.add(Dense(int(input_value_count), input_dim=input_value_count, use_bias=True,
                           kernel_initializer='normal', activation='relu'))

regression_model.add(Dense(1, use_bias=True, kernel_initializer='normal', activation='linear'))
regression_model.compile(loss='mse', optimizer='adam')

print('Training regression network on ' + str(len(np_training_y_SDE)) + ' samples')
regression_history = regression_model.fit(np_training_x_SDE, np_training_y_SDE, epochs=EPOCHS_SDE, validation_split=0,
                                          validation_data=(np_validation_x_SDE, np_validation_y_SDE),
                                          batch_size=BATCH_SIZE, verbose=1, shuffle=True)

# save the regression model
with open('regression_NN.json', 'w') as json_file:
    json_file.write(regression_model.to_json())

# serialize regression weights to HDF5
regression_model.save_weights('regression_NN.h5')

# plot classification network performance
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
fig, ax = plt.subplots(2, 2, figsize=(cm_to_inch(24), cm_to_inch(18)))
fig.tight_layout()

plt.subplot(2, 2, 1)
plt.title('loss and accuracy', **FONT)

x_range = range(1, len(classification_history.history['loss']) + 1)
plt.plot(x_range, classification_history.history['loss'], label='training', color='darkorange')
plt.plot(x_range, classification_history.history['val_loss'], label='val. loss', color='gray')

xmin = int(np.min(x_range)) - 1
xmax = int(np.max(x_range))
plt.xlim(xmin, xmax)
plt.xticks(list(range(xmin, xmax + 1)), **FONT)

ymin = 0
ymax = 1.2
plt.ylim(ymin, ymax)
plt.text((xmax-xmin)*0.05 + xmin, ymax - 0.075*(ymax-ymin), 'a', **FONT_BOLD)
plt.ylabel('crossentropy', **FONT)
plt.xlabel('epoch', **FONT)

if len(classification_history.history['loss']) > 20:
    plt.xticks(list(np.array(list(range(int(len(classification_history.history['loss']) / 10) + 1))) * 10), **FONT)

legend_txt = ['training', 'validation']
plt.legend(legend_txt, loc='upper right')
plt.grid(True)

ax = plt.subplot(2, 2, 3)

ax.plot(x_range, classification_history.history['acc'], color='darkorange')
ax.plot(x_range, classification_history.history['val_acc'], color='gray')
plt.ylabel('categorical accuracy', **FONT)
plt.xlabel('epoch', **FONT)

plt.yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], **FONT)

ymin = 0.5
ymax = 1.1
plt.ylim(ymin, ymax)
xmin = int(np.min(x_range)) - 1
xmax = int(np.max(x_range))
plt.xlim(xmin, xmax)
plt.xticks(list(range(xmin, xmax + 1)), **FONT)
plt.text((xmax-xmin)*0.05 + xmin, ymax - (ymax-ymin)*0.075, 'b', **FONT_BOLD)

if len(classification_history.history['loss']) > 20:
    plt.xticks(list(np.array(list(range(int(len(classification_history.history['loss']) / 10) + 1))) * 10), **FONT)

plt.grid(True)

# create confusion matrix
print('evaluate validation data for confusion matrix')
json_file = open('classification_NN.json', 'r')
classification_model = model_from_json(json_file.read())
json_file.close()
classification_model.load_weights('classification_NN.h5')
classification_result = classification_model.predict([np_validation_x_classes])
confidence_classification_result = [np.max(obj) for obj in classification_result]
classification_result = [np.argmax(y) for y in classification_result]
number_of_classes = len(np.unique(validation_y_classes))
confusion_matrix = np.zeros((number_of_classes, number_of_classes + 1))

for idx in range(len(classification_result)):
    if confidence_classification_result[idx] <= CONFUSION_THRESHOLD:
        confusion_matrix[validation_y_classes[idx], number_of_classes] += 1
    else:
        confusion_matrix[validation_y_classes[idx], classification_result[idx]] += 1

# normalize confusion_matrix row-wise
confusion_matrix_abs = confusion_matrix
confusion_matrix /= confusion_matrix.sum(axis=1)[:, np.newaxis]
print(confusion_matrix)

# plot confusion matrix
ax = plt.subplot(1, 2, 2)
plt.title('confusion matrix', **FONT)
plt.text(-0.5, -0.5, 'c', **FONT_BOLD)
ax.xaxis.tick_top()
ax = sns.heatmap(confusion_matrix, annot=True, fmt='.2f', cmap='Oranges', linewidths=.2, square=True, cbar=False)
ax.yaxis.set_label_position('right')
ax.tick_params(axis=u'both', which=u'both', length=0)
plt.ylabel('actual class', **FONT)
plt.xlabel('predicted class', **FONT)
prediction_labels = class_labels + ['unknown']
plt.yticks(np.arange(number_of_classes) + 0.5, class_labels, rotation=0, **FONT)
plt.xticks(np.arange(number_of_classes + 1) + 0.5, prediction_labels, rotation=90, **FONT)
plt.tight_layout()
ax.set_ylim(number_of_classes, 0)
ax_divider = make_axes_locatable(ax)
cax = ax_divider.append_axes('bottom', size='5%', pad=cm_to_inch(1))
cbar = plt.colorbar(ax.get_children()[0], cax=cax, orientation='horizontal', ticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
cbar.ax.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
fig.savefig('classification_results.pdf', dpi=300)

# save network
json_file = open('regression_NN.json', 'r')
regression_model = model_from_json(json_file.read())
json_file.close()
regression_model.load_weights('regression_NN.h5')

# plot regression (SDE) results model
sns.set_style('darkgrid', {'axes.facecolor': '.9'})
fig, ax = plt.subplots(1, 2, figsize=(cm_to_inch(24), cm_to_inch(12)))
plt.subplot(1, 2, 1)
plt.title('training and validation loss', **FONT)
x_range = range(1, len(regression_history.history['loss']) + 1)
plt.plot(x_range, regression_history.history['loss'], label='training', color='darkorange')
plt.plot(x_range, regression_history.history['val_loss'], label='validation', color='gray')

xmin = int(np.min(x_range)) - 1
xmax = int(np.max(x_range))
plt.xlim(xmin, xmax)
plt.xticks(list(range(xmin, xmax)), **FONT)

ymin = 0
ymax = 0.1
plt.ylim(ymin, ymax)
plt.text((xmax - xmin) * 0.05 + xmin, ymax - 0.075 * (ymax - ymin), 'a', **FONT_BOLD)
plt.ylabel('mean square error (MSE)', **FONT)
plt.xlabel('epoch', **FONT)

if len(regression_history.history['loss']) > 20:
    plt.xticks(list(np.array(list(range(int(len(regression_history.history['loss']) / 10) + 1))) * 10), **FONT)

legend_txt = ['training', 'validation']
plt.legend(legend_txt, loc='upper right')

# benchmark the model
prediction_data = []
for file in sorted(glob.iglob(BENCHMARK_PATH + '*.dat')):
    a = np.loadtxt(file, skiprows=1)
    a /= np.max(abs(a))
    prediction_data.append(a)

if len(prediction_data) > 0:
    prediction_data = np.asarray(prediction_data)
    regression_results = regression_model.predict([prediction_data])

    # plot SDE distribution of valid shapes
    plt.subplot(1, 2, 2)
    plt.title('benchmark on discocytes', **FONT)
    plt.ylabel('probability density', **FONT)
    plt.xlabel('SDE', **FONT)

    SDE_label_list = []
    for idx in range(len(SDE_labels)):
        SDE_label_list.append(
            "{:.2f}\n".format(SDE_labels[idx]) + SDE_full_labels[SDE_labels[idx]])

    # overwrite for customized short labels
    SDE_label_list = []
    SDE_labels_custom = ['SP', 'ST II', 'ST I', 'D', 'E I', 'E II', 'E III']
    for idx in range(len(SDE_labels_custom)):
        SDE_label_list.append(
            "{:.2f}\n".format(SDE_labels[idx]) + SDE_labels_custom[idx])

    plt.xticks(SDE_labels, **FONT)
    ax = plt.gca()
    ax.set_xticklabels(SDE_label_list)
    xmin = np.min(SDE_labels) - 0.25
    xmax = np.max(SDE_labels) + 0.25
    plt.xlim(xmin, xmax)
    pdf, bins = np.histogram(regression_results, bins=np.linspace(-1.25, 1.25, 126), density=True)
    pdf = np.append(pdf, [0])
    ymin = 0
    ymax = 14
    plt.ylim(ymin, ymax)
    plt.text((xmax-xmin)*0.05 + xmin, ymax - 0.075*(ymax-ymin), 'b', **FONT_BOLD)

    plt.bar(bins, pdf, width=bins[1] - bins[0], align='center', color='gray', edgecolor='none', alpha=0.5,
            label='histogram')

    # kernel density function
    x_grid = np.linspace(-1.25, 1.25, 1261)[:, np.newaxis]
    kde = KernelDensity(bandwidth=(bins[1] - bins[0]), kernel='gaussian')
    kde.fit(regression_results)
    log_prob = kde.score_samples(x_grid)
    plt.plot(x_grid, np.exp(log_prob), color='darkorange', linewidth=1.2, label='KDE')
    legend_txt = ['KDE', 'histogram']
    plt.legend(legend_txt, loc='upper right')

    # find and plot peaks
    kde_dist = np.exp(log_prob)
    peaks, _ = find_peaks(kde_dist, height=0, width=[MIN_PEAK_WIDTH, None])
    plt.plot(x_grid[peaks], kde_dist[peaks], 'x', color='darkorange')
    for i in range(len(peaks)):
        plt.text(x_grid[peaks[i]], kde_dist[peaks[i]] + 0.5,
                 "{:.2f} / {:.2f}".format(x_grid[peaks[i]], kde_dist[peaks[i]]), ha='center')

plt.tight_layout()
plt.grid(True)

plt.show()
fig.savefig('regression_results.pdf', dpi=300)
