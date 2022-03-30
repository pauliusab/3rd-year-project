import tensorflow as tf
import numpy as np
import utils as utils
import crossbar_utils as c_utils


class Memristive_Model(tf.keras.Model):

    def __init__(self, name, **kwargs):
        super(Memristive_Model, self).__init__(name=name)


        self.input_layer = tf.keras.layers.Flatten(input_shape=(28, 28), name="input")
        self.hidden_layer = Memristive_Layer(784, 30, activation="sigmoid", name="hidden", **kwargs)
        self.output_layer = Memristive_Layer(30, 10, activation="softmax", name="output", **kwargs)

        self.device_yield = kwargs.get("device_yield", None)
        self.linearity_noise = kwargs.get("linearity_noise", None)
        self.abs_noise = kwargs.get("abs_noise", None)

        # get model directory to store data in
        setup = c_utils.setup_as_str(self.device_yield, self.abs_noise, self.linearity_noise)
        # setup = c_utils.setup_as_str(self.device_yield, self.linearity_noise)
        self.dir = "./data/" + setup + "/" + self.name + "/checkpoints/"
        self.general_dir = "./data/" + setup + "/" + self.name

        c_utils.save_hyperparams(self)


    def call(self, input):
        x = self.input_layer(input)
        x = self.hidden_layer(x)
        x = self.output_layer(x)
        return x


    def train_epoch(self, training_data, n_batches, batch_size, shuffle):

        if (shuffle == True):
            training_data = utils.prepare_training_data(training_data, 1)

        x_train = training_data[0][0]
        y_train = np.expand_dims(training_data[0][1], 1)
        train_array = np.concatenate([x_train, y_train], axis=1)
        batches = np.array(np.split(train_array, int(50000 / batch_size), axis=0))

        # train model for each batch = 1 epoch
        for i in range(n_batches):
            this_x = batches[i, :, :-1]
            this_y = batches[i, :, -1]
            self.train_batch_nonideal(this_x, this_y)


    def train_batch(self, x, y):

        with tf.GradientTape() as tape:
            logits = self(x)
            loss_value = self.loss(y, logits)
        gradients = tape.gradient(loss_value, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))


    def train_batch_nonideal(self, x, y):

        # do a forward pass which uses weights corresponding to real conductances
        with tf.GradientTape() as tape:
            logits = self(x)            # forward pass - calls model on input
            loss_value = self.loss(y, logits)   # calculate loss


        # read weights from conductances and set layer weights to them to be used for gradient calculation
        hidden_layer_w_read = c_utils.readout_g_to_w(self.hidden_layer, self.hidden_layer.conductance_matrix)
        self.hidden_layer.set_weights([hidden_layer_w_read[:-1], hidden_layer_w_read[-1]])

        output_layer_w_read = c_utils.readout_g_to_w(self.output_layer, self.output_layer.conductance_matrix)
        self.output_layer.set_weights([output_layer_w_read[:-1], output_layer_w_read[-1]])

        # get gradient on read weights - this is different than using actual weights
        gradients = tape.gradient(loss_value, self.trainable_weights)       #calculate gradient on weights


        # get learning rate and apply it to gradients
        learning_rate = self.optimizer.get_config().get("learning_rate")
        grad_list = []
        for grad in gradients:
            this_grad = tf.math.scalar_mul(-learning_rate, grad)           #had to add *-1 because the gradient is added later instead of being subtracted
            grad_list.append(this_grad)


        # gather and reformat data
        d_hidden_layer_b = tf.expand_dims(grad_list[1].numpy(), axis=0)
        d_hidden_layer_w = grad_list[0]
        d_hidden_layer = np.asarray(tf.concat([d_hidden_layer_w, d_hidden_layer_b], 0))

        d_output_layer_b = tf.expand_dims(grad_list[3].numpy(), axis=0)
        d_output_layer_w = grad_list[2]
        d_output_layer = np.asarray(tf.concat([d_output_layer_w, d_output_layer_b], 0))


        # apply gradient calculated on read weights to real conductance matrices
        self.hidden_layer.conductance_matrix = self.hidden_layer.conductance_matrix - c_utils.write_w_to_g(self.hidden_layer, d_hidden_layer)
        self.output_layer.conductance_matrix = self.output_layer.conductance_matrix - c_utils.write_w_to_g(self.output_layer, d_output_layer)

        # check for conductance resets
        self.hidden_layer.conductance_matrix = c_utils.check_for_g_reset(self.hidden_layer)
        self.output_layer.conductance_matrix = c_utils.check_for_g_reset(self.output_layer)


    def train_custom(self, training_data, eval_data, **kwargs):

        # get training parameters
        shuffle = kwargs.get("shuffle", False)
        batch_size = kwargs.get("batch_size", 50)
        n_batches = int(kwargs.get("n_batches", 50000 / batch_size))

        save_weights = kwargs.get("save_weights", True)
        #if (save_weights == True):
        save_checkpoints = kwargs.get("save_checkpoints", None)

        epochs = kwargs.get("epochs", None)
        train_full = False
        if (epochs == None):
            train_full = True

        return_history = kwargs.get("return_history", True)
        if (return_history == True):
            acc_history = ([])

        if n_batches > (50000 / batch_size):        # 5000 = training data length
            n_batches = int(50000 / batch_size)
            print("n_batches had to be reduced!")

        if (save_weights == True):
            self.save_weights(self.dir + "before_build")
            utils.save_data([c_utils.layer_to_weights(self.output_layer), c_utils.layer_to_weights(self.hidden_layer)],
                            filename=self.dir + "w_before_build.csv")
            utils.save_conductances([self.output_layer.conductance_matrix, self.hidden_layer.conductance_matrix],
                                    filename=self.dir + "g_before_build.csv")

        # evaluate initial setup
        this_acc = self.evaluate(eval_data[0], eval_data[1], verbose=0)[1]
        print(this_acc)

        # save initial data if needed
        if (return_history == True):
            acc_history.append(this_acc)
        if (save_checkpoints != None):
            c_utils.check_if_cp(self, 0, save_checkpoints)



        if (train_full == False):
            print("**  training for " + str(epochs) + " epochs:  **")

            for e in range(epochs):

                # train the model for 1 epoch
                self.train_epoch(training_data, n_batches, batch_size, shuffle)

                # evaluate accuracy
                this_acc = self.evaluate(eval_data[0], eval_data[1], verbose=0)[1]
                print(this_acc)

                # save data if needed
                if (return_history == True):
                    acc_history.append(this_acc)
                if (save_checkpoints != None):
                    c_utils.check_if_cp(self, e + 1, save_checkpoints)

            # save final accuracy
            if (epochs not in save_checkpoints):
                self.save_weights(self.dir + "epoch_" + str(epochs))
                utils.save_data([c_utils.layer_to_weights(self.output_layer), c_utils.layer_to_weights(self.hidden_layer)],
                                filename=self.dir + "epoch_" + str(epochs) + ".csv")
                utils.save_conductances([self.output_layer.conductance_matrix, self.hidden_layer.conductance_matrix],
                                        filename=self.dir + "g_epoch_" + str(epochs) + ".csv")


        else:
            print("**  training fully  **")
            best_acc = 0
            count = 0
            n_epoch = 0

            s_count1 = False
            s_count3 = False

            while (count < 5):
                n_epoch += 1

                #train the model for 1 epoch
                self.train_epoch(training_data, n_batches, batch_size, shuffle)

                # evaluate accuracy
                this_acc = self.evaluate(eval_data[0], eval_data[1], verbose=0)[1]
                if (this_acc > best_acc):
                    best_config = (self.get_weights(),(self.hidden_layer.conductance_matrix, self.output_layer.conductance_matrix))
                    best_acc = this_acc
                    print(this_acc)
                    count = 0
                else:
                    count += 1
                    print(this_acc, " count: ", count)

                # append accuracy if needed
                if (return_history == True):
                    acc_history.append(this_acc)

                # saving weights
                if (save_weights == True):
                    if (count == 1 and s_count1 == False):
                        s_count1 = c_utils.save_count_cp(self, count, n_epoch, best_config, eval_data)
                    elif (count == 3 and s_count3 == False):
                        s_count3 = c_utils.save_count_cp(self, count, n_epoch, best_config, eval_data)

                    if (save_checkpoints != None):
                        c_utils.check_if_cp(self, n_epoch, save_checkpoints)


            # after full training revert to best_config, save it and confirm final accuracy
            # self.set_weights(best_config)

            # self.save_weights(self.dir + "epoch_" + str(n_epoch) + "_count5")
            # utils.save_data([c_utils.layer_to_weights(self.output_layer), c_utils.layer_to_weights(self.hidden_layer)],
            #                 filename=self.dir + "w_epoch_" + str(n_epoch) + "_count5.csv")
            # utils.save_conductances([self.output_layer.conductance_matrix, self.hidden_layer.conductance_matrix],
            #                         filename=self.dir + "g_epoch_" + str(n_epoch) + "_count5.csv")

            if (save_weights == True):
                self.set_weights(best_config[0])
                self.hidden_layer.conductance_matrix = best_config[1][0]
                self.output_layer.conductance_matrix = best_config[1][1]

                self.save_weights(self.dir + "epoch_" + str(n_epoch) + "_count" + str(count))
                utils.save_data([c_utils.layer_to_weights(self.output_layer), c_utils.layer_to_weights(self.hidden_layer)],
                                filename=self.dir + "w_epoch_" + str(n_epoch) + "_count" + str(count) + ".csv")
                utils.save_conductances([self.output_layer.conductance_matrix, self.hidden_layer.conductance_matrix],
                                        filename=self.dir + "g_epoch_" + str(n_epoch) + "_count" + str(count) + ".csv")


            this_acc = self.evaluate(eval_data[0], eval_data[1], verbose=0)[1]
            print(this_acc)
            if (return_history == True):
                acc_history.append(this_acc)


        if (return_history == True):
            return acc_history



class Memristive_Layer(tf.keras.layers.Layer):

    def __init__(self, n_in, n_out, **kwargs):

        self.device_yield = kwargs.pop("device_yield", None)
        print(self.device_yield)
        self.linearity_noise = kwargs.pop("linearity_noise", None)
        self.abs_noise = kwargs.pop("abs_noise", None)

        self.conductance_range = kwargs.pop("conductance_range", (0.0009972, 0.003514))
        self.w_max = kwargs.pop("w_max", 3)
        self.mapping_scale_f = kwargs.pop("mapping_scale_factor", (self.conductance_range[1] - self.conductance_range[0]) / self.w_max)      # from paper it should be (g_max - g_min / w_max)


        self.ideal_layer = True
        if ((self.device_yield != None) or (self.linearity_noise != None) or (self.abs_noise != None)):
            self.ideal_layer = False

        self.activation = kwargs.pop("activation", "sigmoid")

        super(Memristive_Layer, self).__init__(**kwargs)
        self.n_in = n_in
        self.n_out = n_out

        self.w = self.add_weight(
            shape=(n_in, n_out),
            initializer="random_normal",
            trainable=True,
            name="weights"
        )
        self.b = self.add_weight(
            shape=(n_out,),
            initializer="zeros",
            trainable=True,
            name="biases"
        )

        self.weight_matrix = c_utils.layer_to_weights(self)

        init_g_matrix = self.conductance_range[1] * np.ones((self.weight_matrix.shape[0], self.weight_matrix.shape[1], 2))
        self.conductance_matrix = init_g_matrix - c_utils.w_to_g(self, self.weight_matrix)



    def build(self, input_shape):

        m_size = self.conductance_matrix.shape
        #self.off_devices = np.array([])

        if (self.ideal_layer == False):
            if (self.device_yield != None):
                prob_off = 1 - self.device_yield
                self.off_devices = np.random.choice([0, 1], size=m_size, p=[prob_off, 1 - prob_off])
                self.conductance_matrix = np.where(self.off_devices == 1, self.conductance_matrix, 0)
                self.weight_matrix = c_utils.g_to_w(self, self.conductance_matrix)
                self.set_weights([self.weight_matrix[:-1], self.weight_matrix[-1]])

        else:
            # if there are no nonidealities
            self.weight_matrix = c_utils.layer_to_weights(self)

    def call(self, input):
        # calling is done ideally on conductance values - theres no readout error because the process is physical, not computational
        if (self.ideal_layer == False):
            if (self.device_yield != None):
                self.conductance_matrix = np.where(self.off_devices == 1, self.conductance_matrix, 0)

        weights = c_utils.g_to_w(self, self.conductance_matrix).astype("float32")
        self.set_weights([weights[:-1], weights[-1]])
        
        dot = tf.tensordot(input, self.w, axes=1) + self.b
        if (self.activation == "sigmoid"):
            output = tf.nn.sigmoid(dot)
        elif (self.activation == "softmax"):
            output = tf.nn.softmax(dot)

        return output
