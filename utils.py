import numpy as np
import pandas as pd
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Patch
import math
import os


def normal_disturbance(weights: np.ndarray, mean: float, std: float) -> np.ndarray:
    disturb_matrix = np.random.normal(mean, std, weights.shape)
    disturbed_weights = weights + weights * disturb_matrix / 10
    return disturbed_weights


def get_data(dataset_name: str) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # Preprocess the data (these are NumPy arrays)
        x_train = x_train.reshape(60000, 784).astype("float32") / 255
        x_test = x_test.reshape(10000, 784).astype("float32") / 255

        y_train = y_train.astype("float32")
        y_test = y_test.astype("float32")

        # Reserve 10,000 samples for validation
        x_val = x_train[-10000:]
        y_val = y_train[-10000:]
        x_train = x_train[:-10000]
        y_train = y_train[:-10000]

        # return 2x2 matrix
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def prepare_training_data(training_data, epochs: int):
    x_train = training_data[0]
    y_train = training_data[1]
    y_train = np.expand_dims(y_train, 1)

    prep_data = np.concatenate([x_train, y_train], axis=1)
    training_list = ([])

    for e in range(epochs):
        np.random.shuffle(prep_data)
        x_prep = prep_data[:, :-1]
        y_prep = prep_data[:, -1]
        training_list.append((x_prep, y_prep))
    return training_list


def simulate_exp_data(n_states: int, resistances: list[int]) -> np.ndarray:
    n_states_region = int(n_states / (len(resistances) - 1))
    print(n_states_region)
    output = np.array([])
    for i in range(len(resistances) - 1):
        a = np.array([])
        for n in range(n_states_region):
            b = n * (resistances[i + 1] - resistances[i]) / n_states_region + resistances[i]
            a = np.append(a, b)
        output = np.append(output, a)
    return output


def read_data(filename: str) -> np.ndarray:
    data = pd.read_csv(filename,
                       header=None)  # header = none ensures that the first row is not used as the collumn names
    # print(data)
    data_array = data.to_numpy()  # ignores the header row
    # print(data_array)
    return data_array


def save_data(data, filename: str, **kwargs):
    header = kwargs.get("header", None)
    gaps = kwargs.get("gaps", False)
    dir = kwargs.get("dir", None)
    cwd = os.getcwd()

    f_dir = filename

    if (dir != None):
        dir = os.path.join(cwd, dir)
        f_dir = dir + "/" + f_dir
        if not os.path.exists(dir):
            os.makedirs(dir)
    else:
        dir = cwd

    print("save location: " + f_dir)

    f = open(f_dir, "w")
    writer = csv.writer(f)

    if (header != None):
        writer.writerow(header)

    n_rows = []
    n_colls = []
    single_coll = False

    for a in data:
        try:
            n_rows.append(np.asarray(a, dtype=object).shape[0])
        except IndexError:
            n_rows = [len(data)]
            single_coll = True
        try:
            n_colls.append(np.asarray(a, dtype=object).shape[1])
        except IndexError:
            n_colls.append(1)
        except AttributeError:
            n_colls.append(1)

    for i in range(max(n_rows)):
        row = []
        if (single_coll == False):
            for n in range(len(data)):
                try:
                    asd = data[n][i]
                    if(data[n][i] == ""):
                        row.extend(None)
                    else:
                        row.extend(data[n][i])
                except IndexError:
                    for x in range(n_colls[n]):
                        row.append(None)
                except TypeError:
                    row.append(data[n][i])
                if (gaps == True):
                    row.append('')
        else:
            try:
                row.extend(data[i])
            except TypeError:
                row.append(data[i])

        writer.writerow(row)

    f.close()


def save_conductances(data, filename: str, **kwargs):
    header = kwargs.get("header", None)
    gaps = kwargs.get("gaps", False)
    dir = kwargs.get("dir", None)
    cwd = os.getcwd()

    f_dir = filename

    if (dir != None):
        dir = os.path.join(cwd, dir)
        f_dir = dir + "/" + f_dir
        if not os.path.exists(dir):
            os.makedirs(dir)
    else:
        dir = cwd

    print("save location: " + f_dir)

    f = open(f_dir, "w")
    writer = csv.writer(f)

    if (header != None):
        writer.writerow(header)

    n_rows = []
    n_colls = []
    single_coll = False

    for a in data:
        try:
            n_rows.append(np.asarray(a).shape[0])
        except IndexError:
            n_rows = [len(data)]
            single_coll = True
        try:
            n_colls.append(np.asarray(a).shape[1] * 2)
        except IndexError:
            n_colls.append(1)
        except AttributeError:
            n_colls.append(1)

    for i in range(max(n_rows)):
        row = []

        for n in range(len(data)):
            try:
                asd = np.asarray(data[n][i]).shape[0]
                for x in range(np.asarray(data[n][i]).shape[0]):
                    row.extend(data[n][i][x])
            except IndexError:
                for x in range(n_colls[n]):
                    row.append(None)
            except TypeError:
                row.append(data[n][i])
            if (gaps == True):
                row.append('')

        writer.writerow(row)

    f.close()


def plot_weights(weights, names, **kwargs):
    zoom = kwargs.get("zoom", None)
    if (zoom != None):
        if (zoom < 1):
            print("Zoom value too small!")
            zoom = None
    zero_color = kwargs.get("zero_c", "white")
    title = kwargs.get("title", None)
    print(zoom)

    newcmp = create_cmp(zero_color)

    weights_zoomed = ([])
    weights_arr = ([])
    if names != None:
        name_arr = ([])
    else:
        name_arr = None

    n = len(weights)
    n_rows = 0
    n_colls_arr = ([])
    n_colls = 0
    n_colls_row = 0
    if (n < 20):
        for i in range(n):
            n_rows += 1
            if (n > 1):

                if (len(weights[i]) < 20):
                    n_colls_row = 0
                    for x in range(len(weights[i])):
                        new_shape = find_arr_size(weights[i][x].shape[0], weights[i][x].shape[1])
                        new_shape = [int(new_shape[s]) for s in range(len(new_shape))]
                        weights[i][x] = np.reshape(weights[i][x], new_shape)
                        weights[i][x] = np.ma.masked_equal(weights[i][x], 0)

                        if (zoom != None):
                            weights_zoomed.append(weights[i][x][
                                                  int((new_shape[0] - new_shape[0] / zoom) / 2): int(
                                                      new_shape[0] - ((new_shape[0] - new_shape[0] / zoom) / 2)),
                                                  int((new_shape[1] - new_shape[1] / zoom) / 2): int(
                                                      new_shape[1] - ((new_shape[1] - new_shape[1] / zoom) / 2))
                                                  ])
                        weights_arr.append((weights[i][x]))
                        if names != None:
                            name_arr.append((names[i][x]))
                        n_colls_row += 1

                else:
                    new_shape = find_arr_size(weights[i].shape[0], weights[i].shape[1])
                    new_shape = [int(new_shape[s]) for s in range(len(new_shape))]
                    weights[i] = np.reshape(weights[i], new_shape)
                    weights[i] = np.ma.masked_equal(weights[i], 0)

                    if (zoom != None):
                        weights_zoomed.append(weights[i][
                                              int((new_shape[0] - new_shape[0] / zoom) / 2): int(
                                                  new_shape[0] - ((new_shape[0] - new_shape[0] / zoom) / 2)),
                                              int((new_shape[1] - new_shape[1] / zoom) / 2): int(
                                                  new_shape[1] - ((new_shape[1] - new_shape[1] / zoom) / 2))
                                              ])
                    weights_arr.append((weights[i]))
                    if names != None:
                        name_arr.append(names[i])

                    n_rows = 1
                    n_colls_row += 1

            n_colls_arr.append(n_colls_row)

        n_colls = np.amax(n_colls_arr)

    else:
        new_shape = find_arr_size(weights.shape[0], weights.shape[1])
        new_shape = [int(new_shape[s]) for s in range(len(new_shape))]
        weights = np.reshape(weights, new_shape)
        weights = np.ma.masked_equal(weights, 0)

        if (zoom != None):
            weights_zoomed.append(weights[
                                  int((new_shape[0] - new_shape[0] / zoom) / 2): int(
                                      new_shape[0] - ((new_shape[0] - new_shape[0] / zoom) / 2)),
                                  int((new_shape[1] - new_shape[1] / zoom) / 2): int(
                                      new_shape[1] - ((new_shape[1] - new_shape[1] / zoom) / 2))
                                  ])
        weights_arr.append((weights))
        if names != None:
            name_arr.append(names)

        n_rows = 1
        n_colls = 1

    # min = np.amin(np.asarray(weights_arr))
    # max = np.amax(np.asarray(weights_arr))
    # vmax = np.amax([abs(min), max])

    min = ([])
    max = ([])

    for w in weights_arr:
        min.append(np.amin(w))
        max.append(np.amax(w))

    vmax = np.amax([abs(np.asarray(min)), np.asarray(max)])

    print("vmax: ", vmax)

    if (zoom != None):
        plot_all_data(weights_zoomed, n_rows, n_colls, newcmp, vmax, title, name_arr, zero_color)
    else:
        plot_all_data(weights_arr, n_rows, n_colls, newcmp, vmax, title, name_arr, zero_color)


def plot_all_data(data, n_rows, n_colls, cmap, vmax, fig_title, plot_names, zeroes):
    n = len(data)
    scale_f_vertical = n_rows
    scale_f_horizontal = n_colls

    fig_width = 5 * n_colls
    if (5 * n_rows > 10):
        fig_height = 12
    else:
        fig_height = 5 * n_rows

    fig = plt.figure(figsize=(fig_width, fig_height))

    if (fig_title != None):
        fig.suptitle(fig_title, fontsize=16)

    row_index = 0
    collumn_index = 0
    for i in range(n):
        if (i % n_colls == 0 and i != 0):
            collumn_index = 0
            row_index += 1
        elif (i != 0):
            collumn_index += 1

        ax = fig.add_subplot(n_rows, n_colls, i + 1)  # 1x2 grid, 1st element
        if (plot_names != None):
            ax.title.set_text(plot_names[i])

        if (fig_height <= 10):
            v_gaps = 0.1 / scale_f_vertical
        else:
            v_gaps = 0.05
        h_gaps = 0.1 / scale_f_horizontal
        if (fig_height <= 10):
            top_gap = 0.2 / scale_f_vertical
        else:
            top_gap = 0.1
        right_gap = 0.2 / scale_f_horizontal

        width = (1 - (right_gap + n_colls * h_gaps)) / n_colls
        height = (1 - (top_gap + n_rows * v_gaps)) / n_rows
        # print("parameters: ", v_gaps, h_gaps, top_gap, right_gap, width, height)

        left = h_gaps + (h_gaps + width) * collumn_index
        bottom = v_gaps + (v_gaps + height) * (n_rows - row_index - 1)

        # print(i)
        # print(collumn_index, row_index)
        # print(left, bottom, width, height)

        ax.set_position([left, bottom, width, height])

        psm = ax.pcolormesh(data[i], cmap=cmap, vmin=-1 * vmax, vmax=vmax)
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False
        )

        if (collumn_index == n_colls - 1):
            cax = fig.add_axes([left + width + (0.02 / scale_f_horizontal),
                                bottom,
                                0.04 / scale_f_horizontal,
                                height
                                ])
            fig.colorbar(psm, cax=cax)

    if (zeroes != "white"):
        legend_elements = [Patch(facecolor=cmap(np.nan), label='Zero conductance')]
        fig.legend(handles=legend_elements, loc=8)

    plt.show()


def create_cmp(zero_color):
    top = cm.get_cmap('Oranges', 256)
    bottom = cm.get_cmap('Blues_r', 256)

    newcolors = np.vstack((bottom(np.linspace(0.3, 1, 256)),
                           top(np.linspace(0, 0.7, 256))))
    cmp = ListedColormap(newcolors, name='OrangeBlue')
    if (zero_color != None):
        cmp.set_bad(zero_color)

    return cmp


def find_arr_size(shape0, shape1):
    x = shape0 * shape1

    # if it is already square
    if (math.sqrt(x) == shape0):
        return (shape0, shape1)

    else:
        max_axis = max([shape0, shape1])
        new_s0 = shape0
        new_s1 = shape1
        for i in range(2, max_axis):
            if (max_axis % i == 0):
                if (shape0 > shape1):
                    s0 = shape0 / i
                    s1 = shape1 * i
                elif (shape0 < shape1):
                    s0 = shape0 * i
                    s1 = shape1 / i

                # check if it is more square-like
                if (s0 + s1 < shape0 + shape1):
                    if (s0 + s1 < new_s0 + new_s1):
                        new_s0 = s0
                        new_s1 = s1
                    else:
                        print("new shape: ", new_s0, new_s1)
                        return (new_s0, new_s1)

        print("new shape: ", new_s0, new_s1)
        return (new_s0, new_s1)


def get_acc_analysis_new(folder, **kwargs):
    dirs = os.listdir("./data/" + folder)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    dirs.sort(key=lambda x: int(x[5:]))     # sort by model nr
    all_eval = []
    params_set = []

    for model in dirs:
        if os.path.exists("./data/" + folder + "/" + model + "/parameters.txt"):
            with open("./data/" + folder + "/" + model + "/parameters.txt") as f:
                params = f.readlines()
        else:
            params = ""

        data = pd.read_csv("./data/" + folder + "/" + model + "/accuracy_history.csv",
                           header=None)  # header = none ensures that the first row is not used as the collumn names
        data_array = data.to_numpy()  # ignores the header row
        eval_history = data_array[:, 0]
        test_acc = data_array[0, 1]

        # this is used to add test_acc in the first row
        eval_history = list(eval_history)

        eval_history.insert(0, "")
        eval_history.insert(0, test_acc)
        eval_history.insert(0, params)
        all_eval.extend([eval_history])

    all_eval.sort(key=lambda x: str(x[0]))      # sort by parameters


    this_mean_arr = []
    means = []
    i = 0
    x = len(all_eval)
    this_param = all_eval[0][0]     #read the first parameter
    while i < x:
        print(all_eval[i][0])
        if all_eval[i][0] not in params_set:
            if not this_mean_arr:       #if the list is empty. needed for the first loop
                print(all_eval[i][1])
                this_mean_arr.append(all_eval[i][1])
            else:
                asd = [[this_param, ["mean"], sum(this_mean_arr) / len(this_mean_arr)]]
                means.extend(asd)
                this_mean_arr = []
                print(all_eval[i][1])
                this_mean_arr.append(all_eval[i][1])

            params_set.append(all_eval[i][0])
            this_param = all_eval[i][0]

        else:
            all_eval[i][0] = ""
            print(all_eval[i][1])
            this_mean_arr.append(all_eval[i][1])
        i += 1

    all_eval.append([""])
    for a in means:
        all_eval.append(a)
    print(all_eval)

    return all_eval

def get_acc_analysis_noise(folder, **kwargs):
    dirs = os.listdir("./data/" + folder)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    dirs.sort(key=lambda x: x[10:11])     # sort by model nr
    all_eval = []
    params_set = []

    for model in dirs:
        if os.path.exists("./data/" + folder + "/" + model + "/model 0/parameters.txt"):
            with open("./data/" + folder + "/" + model + "/model 0/parameters.txt") as f:
                params = f.readlines()
        else:
            params = ""

        data = pd.read_csv("./data/" + folder + "/" + model + "/model 0/accuracy_history.csv",
                           header=None)  # header = none ensures that the first row is not used as the collumn names
        data_array = data.to_numpy()  # ignores the header row
        eval_history = data_array[:, 0]
        test_acc = data_array[0, 1]

        # this is used to add test_acc in the first row
        eval_history = list(eval_history)

        eval_history.insert(0, "")
        eval_history.insert(0, test_acc)
        eval_history.insert(0, params)
        all_eval.extend([eval_history])

    all_eval.sort(key=lambda x: str(x[0]))      # sort by parameters


    this_mean_arr = []
    means = []
    i = 0
    x = len(all_eval)
    this_param = all_eval[0][0]     #read the first parameter
    while i < x:
        print(all_eval[i][0])
        if all_eval[i][0] not in params_set:
            if not this_mean_arr:       #if the list is empty. needed for the first loop
                print(all_eval[i][1])
                this_mean_arr.append(all_eval[i][1])
            else:
                asd = [[this_param, ["mean"], sum(this_mean_arr) / len(this_mean_arr)]]
                means.extend(asd)
                this_mean_arr = []
                print(all_eval[i][1])
                this_mean_arr.append(all_eval[i][1])

            params_set.append(all_eval[i][0])
            this_param = all_eval[i][0]

        else:
            all_eval[i][0] = ""
            print(all_eval[i][1])
            this_mean_arr.append(all_eval[i][1])
        i += 1

    all_eval.append([""])
    for a in means:
        all_eval.append(a)
    print(all_eval)

    return all_eval

def get_acc_analysis_old(folder, **kwargs):
    yield_interval = kwargs.get("yield_interval", None)
    if yield_interval != None:
        yield_increment = kwargs.get("yield_increment", None)
        init_yield = kwargs.get("init_yield", None)

    w_max_interval = kwargs.get("w_max_interval", None)
    if w_max_interval != None:
        w_max_increment = kwargs.get("w_max_increment", None)
        init_w_max = kwargs.get("init_w_max", None)

    dirs = os.listdir("./data/" + folder)
    if ".DS_Store" in dirs:
        dirs.remove(".DS_Store")
    dirs.sort(key=lambda x: int(x[5:]))     # sort by model nr
    all_eval = []
    params_set = []

    for model in dirs:
        if os.path.exists("./data/" + folder + "/" + model + "/parameters.txt"):
            with open("./data/" + folder + "/" + model + "/parameters.txt") as f:
                params = f.readlines()
        else:
            params = ""

        data = pd.read_csv("./data/" + folder + "/" + model + "/accuracy/accuracy_history.csv",
                           header=None)  # header = none ensures that the first row is not used as the collumn names
        data_array = data.to_numpy()  # ignores the header row
        eval_history = data_array[:, 0]
        test_acc = data_array[0, 1]

        # this is used to add test_acc in the first row
        eval_history = list(eval_history)

        eval_history.insert(0, "")
        eval_history.insert(0, test_acc)
        eval_history.insert(0, params)
        all_eval.extend([eval_history])

    yield_row = []
    w_max_row = []

    if yield_interval == None and w_max_interval == None:
        all_eval.sort(key=lambda x: str(x[0]))      # sort by parameters

    elif yield_interval != None and w_max_interval == None:
        for x in range(len(all_eval[:, 2])):
            yield_row.append(["yield = " + str(init_yield + x * yield_increment)])
            for i in range(yield_interval - 1):
                yield_row.append("")
        all_eval[:, 0] = yield_row

    elif yield_interval == None and w_max_interval != None:
        for x in range(len(all_eval[:, 2])):
            w_max_row.append(["w_max = " + str(init_w_max + x * w_max_increment)])
            for i in range(w_max_interval - 1):
                w_max_row.append("")
        all_eval[:, 0] = w_max_row

    # not finished !!!
    # elif yield_interval != None and w_max_interval != None:
    #     for x in range(len(all_eval[:, 2])):
    #         yield_row.append(["yield = " + str(init_yield + x * yield_increment) + "w_max = " + str(init_w_max + x * w_max_increment)])
    #         for i in range(yield_interval - 1):
    #             yield_row.append("")

    this_mean_arr = []
    means = []
    i = 0
    x = len(all_eval)
    this_param = all_eval[0][0]     #read the first parameter
    while i < x:
        print(all_eval[i][0])
        if all_eval[i][0] not in params_set:
            if not this_mean_arr:       #if the list is empty. needed for the first loop
                print(all_eval[i][1])
                this_mean_arr.append(all_eval[i][1])
            else:
                asd = [[this_param, ["mean"], sum(this_mean_arr) / len(this_mean_arr)]]
                means.extend(asd)
                this_mean_arr = []
                print(all_eval[i][1])
                this_mean_arr.append(all_eval[i][1])

            params_set.append(all_eval[i][0])
            this_param = all_eval[i][0]

        else:
            all_eval[i][0] = ""
            print(all_eval[i][1])
            this_mean_arr.append(all_eval[i][1])
        i += 1

    all_eval.append([""])
    for a in means:
        all_eval.append(a)
    print(all_eval)


    return all_eval

def load_saved_weights(dir):  # "./data/ideal/model 0/checkpoints/w_epoch_0.csv"
    weights = pd.read_csv(dir, header=None)  # header = none ensures that the first row is not used as the collumn names
    weight_array = weights.to_numpy()  # ignores the header row
    output_l_w = weight_array[:31, :10]
    hidden_l_w = weight_array[:, 10:40]
    return output_l_w, hidden_l_w


def load_saved_g(dir):  # "./data/ideal/model 0/checkpoints/w_epoch_0.csv"
    weights = pd.read_csv(dir, header=None)  # header = none ensures that the first row is not used as the collumn names
    weight_array = weights.to_numpy()  # ignores the header row
    output_l_g = weight_array[:31, :20]
    hidden_l_g = weight_array[:, 20:80]
    return output_l_g, hidden_l_g
