import utils as utils
import crossbar as crossbar
import numpy as np
import tensorflow as tf
import os

def create_model(name, **kwargs):
    model = crossbar.Memristive_Model(str(name), **kwargs)
    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        run_eagerly=True,
        metrics=["accuracy"]
    )

    return model

def pick_name(setup, name, index):
    cwd = os.getcwd()
    cwd = cwd + "/data/" + setup + "/"
    if not os.path.exists(cwd):
        os.makedirs(cwd)


    try_name = name + str(index)
    dir = os.path.join(cwd, try_name)


    if os.path.exists(dir):
        if not os.listdir(dir):
            print(try_name, " folder empty")
        else:
            index += 1
            try_name = pick_name(setup, name, index)
    else:
        os.mkdir(dir)

    #print(dir)
    return try_name

def setup_as_str(device_yield, abs_noise, linearity_noise):
    setup = ""
    if (device_yield == None and abs_noise == None and linearity_noise == None):
        setup = "ideal"
    if (device_yield != None):
        setup = setup + " yield=" + str(device_yield)
    if (abs_noise != None):
        setup = setup + " absNoise=" + str(abs_noise)
    if (linearity_noise != None):
        setup = setup + " linNoise=" + str(linearity_noise)

    return setup

def check_if_cp(model, epoch, save_checkpoints):
    if (epoch in save_checkpoints):
        model.save_weights(model.dir + "epoch_" + str(epoch))
        utils.save_data([layer_to_weights(model.output_layer), layer_to_weights(model.hidden_layer)],
                        filename=model.dir + "w_epoch_" + str(epoch) + ".csv")
        utils.save_conductances([model.output_layer.conductance_matrix, model.hidden_layer.conductance_matrix],
                        filename=model.dir + "g_epoch_" + str(epoch) + ".csv")

def save_count_cp(model, count, epoch, best_config, eval_data):
    this_config = (model.get_weights(), (model.hidden_layer.conductance_matrix, model.output_layer.conductance_matrix))
    print("return to best config")
    model.set_weights(best_config[0])
    model.hidden_layer.conductance_matrix = best_config[1][0]
    model.output_layer.conductance_matrix = best_config[1][1]

    model.evaluate(eval_data[0], eval_data[1], verbose=1)
    print("save data")
    model.save_weights(model.dir + "epoch_" + str(epoch) + "_count" + str(count))
    utils.save_data([layer_to_weights(model.output_layer), layer_to_weights(model.hidden_layer)],
                    filename=model.dir + "w_epoch_" + str(epoch) + "_count" + str(count) + ".csv")
    utils.save_conductances([model.output_layer.conductance_matrix, model.hidden_layer.conductance_matrix],
                    filename=model.dir + "g_epoch_" + str(epoch) + "_count" + str(count) + ".csv")


    print("return to previous config")
    model.set_weights(this_config[0])
    model.hidden_layer.conductance_matrix = this_config[1][0]
    model.output_layer.conductance_matrix = this_config[1][1]
    model.evaluate(eval_data[0], eval_data[1], verbose=1)

    return True

def layer_to_weights(layer) -> np.ndarray:
    b = tf.expand_dims(layer.b.numpy(), axis=0)
    w = layer.w.numpy()
    l_weights = np.asarray(tf.concat([w, b], 0))
    return l_weights

# this function is used whenever weight update needs to be written to conductances
def write_w_to_g(layer, d_w):
    g_min = layer.conductance_range[0]
    g_max = layer.conductance_range[1]
    scale_f = layer.mapping_scale_f

    flat_w = np.reshape(d_w, (layer.weight_matrix.shape[0] * layer.weight_matrix.shape[1], 1))

    # calculate the change in conductances
    d_g_matrix = np.where(flat_w >= 0, flat_w * [[0, scale_f]], -flat_w * [[scale_f, 0]])
    #d_g_matrix = d_g_matrix * scale_f

    # reshape and adjust conductance matrix
    g_matrix = np.reshape(d_g_matrix, (layer.weight_matrix.shape[0], layer.weight_matrix.shape[1], 2))

    # return the change in conductances that corresponds to d_w
    return g_matrix

# This function is used to ideally convert between w and g. used during initialization
def w_to_g(layer, d_w):
    g_min = layer.conductance_range[0]
    g_max = layer.conductance_range[1]
    scale_f = layer.mapping_scale_f

    flat_w = np.reshape(d_w, (layer.weight_matrix.shape[0] * layer.weight_matrix.shape[1], 1))

    # calculate the change in conductances
    d_g_matrix = np.where(flat_w >= 0, flat_w * [[0, scale_f]], -flat_w * [[scale_f, 0]])
    #d_g_matrix = d_g_matrix * scale_f

    # reshape and adjust conductance matrix
    g_matrix = np.reshape(d_g_matrix, (layer.weight_matrix.shape[0], layer.weight_matrix.shape[1], 2))

    # return the change in conductances that corresponds to d_w
    return g_matrix

# used in call function to convert real conductances to weights
def g_to_w(layer, g_matrix):
    g_min = layer.conductance_range[0]
    g_max = layer.conductance_range[1]

    # get dot product to calculate effective conductance from the pair.
    # This removes the ,2 dimension at the end of conductance matrix and subtracts negative from positive conductances
    g_eff = np.dot(g_matrix, np.array([1, -1]))

    # convert conductances to weights
    weights = g_eff /layer.mapping_scale_f
    return weights

# used during training
def readout_g_to_w(layer, g_matrix):
    g_min = layer.conductance_range[0]
    g_max = layer.conductance_range[1]

    # if (layer.abs_noise != None):
    #     disturb = np.random.normal(loc=0.0, scale=layer.abs_noise, size=g_matrix.shape)
    #     #print(disturb)
    #     g_matrix = g_matrix + disturb

    if (layer.abs_noise != None):
        disturb_map = np.random.choice([0, 1], size=g_matrix.shape, p=[0.2, 0.8])
        disturb = np.random.lognormal(mean=layer.abs_noise, sigma=layer.linearity_noise, size=g_matrix.shape) / 100
        g_matrix = np.where(disturb_map == 1, g_matrix * (1 - disturb), g_matrix)

    # get dot product to calculate effective conductance from the pair.
    # This removes the ,2 dimension at the end of conductance matrix and subtracts negative from positive conductances
    g_eff = np.dot(g_matrix, np.array([1, -1]))

    # convert conductances to weights
    weights = g_eff / layer.mapping_scale_f



    return weights

def check_for_g_reset(layer):
    g_min = layer.conductance_range[0]
    g_max = layer.conductance_range[1]
    scale_f = layer.mapping_scale_f

    if(layer.ideal_layer == False and layer.device_yield != None):
        g_reset_map = np.where((layer.conductance_matrix < g_min) & (layer.off_devices == 1), 1, 0)
    else:
        g_reset_map = np.where(layer.conductance_matrix < g_min, 1, 0)

    sum = np.sum(g_reset_map)
    if(np.sum(g_reset_map) != 0):
        #print("reset")
        # get matrix of weight values that need to be reset
        w_reset_map = np.dot(g_reset_map, np.array([1, 1]))
        flat_w_map = np.reshape(w_reset_map, (w_reset_map.shape[0] * w_reset_map.shape[1], 1))
        new_g_map_flat = np.where(flat_w_map == 1, flat_w_map * [[1, 1]], flat_w_map * [[0, 0]])
        new_g_map = np.reshape(new_g_map_flat, (layer.weight_matrix.shape[0], layer.weight_matrix.shape[1], 2))


        w_matrix = layer_to_weights(layer)
        w_reset_matrix = np.where(w_reset_map > 0, w_matrix, 0)
        flat_w = np.reshape(w_reset_matrix, (layer.weight_matrix.shape[0] * layer.weight_matrix.shape[1], 1))

        # calculate the change in conductances
        d_g_matrix = np.where(flat_w >= 0, flat_w * [[0, scale_f]], -flat_w * [[scale_f, 0]])
        #d_g_matrix = d_g_matrix * scale_f
        # fill out the opposite conductances
        reset_g_matrix = new_g_map_flat * g_max - d_g_matrix

        # reshape and adjust conductance matrix
        g_reset_matrix = np.reshape(reset_g_matrix, (layer.weight_matrix.shape[0], layer.weight_matrix.shape[1], 2))

        # converge the the reset elements with the old ones using g_map
        new_g_matrix = np.where(new_g_map == 1, g_reset_matrix, layer.conductance_matrix)
    else:
        new_g_matrix = layer.conductance_matrix

    return new_g_matrix

def save_hyperparams(model):
    #params = "device yield = " + str(model.device_yield) + " conductance range = " + str(model.hidden_layer.conductance_range) + " w_max = " + str(model.hidden_layer.w_max)

    params = "abs_noise = " + str(model.abs_noise) + "w_max = " + str(model.hidden_layer.w_max)
    print(params)

    cwd = os.getcwd()
    dir = model.general_dir
    dir = os.path.join(cwd, dir)

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(dir + "/parameters.txt", 'w') as f:
        f.write(params)