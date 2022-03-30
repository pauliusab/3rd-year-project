import utils as utils
import crossbar_utils as c_utils




def train_models(n_models, training_data, eval_data, test_data, **kwargs):


    equalize_init = kwargs.pop("equalize_init", False)
    save_acc = kwargs.pop("save_acc", True)

    device_yield = kwargs.get("device_yield", None)
    abs_noise = kwargs.get("abs_noise", None)
    linearity_noise = kwargs.get("linearity_noise", None)
    #scaling_factor = kwargs.get("mapping_scale_f", 0.3)
    w_max = kwargs.get("w_max", 3)

    setup = c_utils.setup_as_str(device_yield, abs_noise, linearity_noise)

    print(setup)

    for i in range(n_models):
        print("n: ", i)
        name = c_utils.pick_name(setup, "model ", i)


        model = c_utils.create_model(
            name,
            device_yield=device_yield,
            abs_noise=abs_noise,
            linearity_noise=linearity_noise,
            w_max=w_max,
            #mapping_scale_factor=scaling_factor
        )

        # print(model.loss)
        # print(model.trainable_weights)
        # print(model.optimizer)

        if(i == 0 and equalize_init == True):
            hidden_layer_w = model.hidden_layer.get_weights()
            output_layer_w = model.output_layer.get_weights()
        elif(i != 0 and equalize_init == True):
            model.hidden_layer.set_weights(hidden_layer_w)
            model.output_layer.set_weights(output_layer_w)

        eval_history = model.train_custom(training_data=training_data,
                                       eval_data=eval_data,
                                       test_data=test_data,
                                       **kwargs
                                       )


        print("\n" + "final test accuracy: ")
        final_acc = model.evaluate(test_data[0], test_data[1], verbose=1)[1]

        if(save_acc == True):
            utils.save_data([eval_history, [final_acc]], "accuracy_history.csv", gaps=False, dir="data/" + setup + "/" + name)

def weight_map(training_data, eval_data, test_data, **kwargs):

    epochs = kwargs.pop("epochs", None)
    return_history = kwargs.pop("return_history", True)
    shuffle = kwargs.pop("shuffle", False)
    batch_size = kwargs.pop("batch_size", 50)
    n_batches = int(kwargs.pop("n_batches", 50000 / batch_size))

    model = c_utils.create_model("asd", **kwargs)
    model.evaluate(test_data[0], test_data[1], verbose=1)


    w_hl_1 = model.hidden_layer.weight_matrix
    w_ol_1 = model.output_layer.weight_matrix

    acc = model.train_custom(training_data=training_data,
                             eval_data=eval_data,
                             test_data=test_data,
                             epochs=epochs,
                             return_history=return_history,
                             shuffle=shuffle,
                             batch_size=batch_size,
                             n_batches=n_batches,
                             **kwargs)

    model.evaluate(test_data[0], test_data[1], verbose=1)


    w_hl_2 = model.hidden_layer.weight_matrix
    w_ol_2 = model.output_layer.weight_matrix

    dw_hl = w_hl_2 - w_hl_1
    dw_ol = w_ol_2 - w_ol_1


    # utils.plot_weights([[w_hl_1, w_hl_2], [w_ol_1, w_ol_2], [dw_hl, dw_ol]],
    #                    [["w_hl_1", "w_hl_2"], ["w_ol_1", "w_ol_2"], ["dw_hl", "dw_ol"]],
    #                    title="weight maps", zero_c="black")

    return acc



mnist_data = utils.get_data("mnist")
training_data = mnist_data[0]
eval_data = mnist_data[1]
test_data = mnist_data[2]


train_models(1, training_data, eval_data, test_data,
                     equalize_init=False,
                     shuffle=True,
                     abs_noise=3,
                     linearity_noise=1,
                     w_max=3,
                     save_weights=False,
                     save_checkpoints=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300],
                     save_acc=False
             )


