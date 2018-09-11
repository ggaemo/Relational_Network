
import tensorflow as tf
import ops

class RelationalNetwork():

    def __init__(self, inputs, idx_to_value,
                 word_embedding_size, g_theta_layers, f_phi_layers, img_encoding_layers,
                 rnn_hidden_dim):

        self.g_theta_layers = g_theta_layers
        self.f_phi_layers = f_phi_layers
        self.encoding_layers = img_encoding_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.is_training = tf.placeholder(tf.bool, shape=None)



        def build_mlp(inputs, layers):
            print('build mlp')
            outputs = list()
            outputs.append(inputs)

            for layer_dim in layers:
                layer_input = outputs[-1]
                fc_output = tf.layers.dense(layer_input, layer_dim, activation=tf.nn.relu)
                # bn_output = tf.layers.batch_normalization(fc_output,
                #                                          training =is_train)
                                                         # updates_collections=None) #decay
                # 0.99 or 0.95 or 0.90
                print(fc_output.shape)
                outputs.append(fc_output)
            return outputs[-1]

        def build_conv(input, layer_config):
            print('build convnet')
            for layer_num, layer_config in enumerate(layer_config):
                (num_filter, kernel_size, stride) = layer_config
                with tf.variable_scope('conv_layer_{}'.format(layer_num)):
                    input = ops.conv(input, num_filter, kernel_size, stride, 'ln',
                                     tf.nn.relu, self.is_training)
                    print(input.shape)
            return input


        def get_embedding_variable(var_name, embedding_size, inputs):
            vocab_size = len(idx_to_value[var_name])
            with tf.variable_scope('embedding_layer'):
                variable_embeddings = tf.get_variable(name='variable_embeddings_{}'.format(var_name),
                                                      shape=[vocab_size, embedding_size],
                                                      initializer=tf.random_uniform_initializer(-1, 1))

                embed_variable = tf.nn.embedding_lookup(variable_embeddings, inputs,
                                                        name='variable_lookup_{}'.format(var_name))
            return embed_variable

        img = inputs['img']

        img.set_shape([None, 128, 128, 3])
        print('img set shape at 128 128 3')
        ans = inputs['ans']
        qst = inputs['qst']
        qst_len = tf.squeeze(inputs['qst_len'], axis=1)

        _, height, width, num_input_channel = img.get_shape().as_list()
        batch_size = tf.shape(img)[0]

        # do this if set_shape is not done

        # height = tf.shape(img)[1]
        # width = tf.shape(img)[2]
        # num_input_channel = tf.shape(img)[-1]

        question_embed = get_embedding_variable('question', word_embedding_size, qst)

        with tf.variable_scope('question_embedding'):
            rnn_cell = tf.contrib.rnn.LSTMCell(num_units=self.rnn_hidden_dim)
            rnn_outputs, last_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                         inputs=question_embed,
                                                         dtype=tf.float32,
                                                         sequence_length=qst_len,
                                                         parallel_iterations=71

                                                     )

        #     qst_len_index_by_batch = tf.stack(
        #         [tf.range(qst_len.get_shape()[0],dtype=tf.int64),
        #          qst_len], axis=1)
        #     # for each instance, get the last hidden vector (this is needed because the
        #     # length of each question is different and padding of 0 occurs with batching
        #
        #     question_embedding = tf.gather_nd(outputs,qst_len_index_by_batch)
        #     # tf.gather_nd, indices defines slices into the first N dimensions of params, where N = indices.shape[-1].
        #     question_embedding = tf.expand_dims(question_embedding, axis=1) # to make
        #     # this as a sequence of length 1.
        #     # question_embedding.shape = (batch_size, 1, question_word_embedding_size)
        #
        # mask = tf.sequence_mask(num_pair, max_num_pair, dtype=tf.float32)
        # mask = tf.expand_dims(mask, axis=2)
        # question_embedding = tf.multiply(question_embedding, mask) # supports broadcasting

        #TODO check if dynamic_rnn gets the last state according to the sequence_length
        # input given

        # encoded_qst = rnn_outputs[:, -1, :] # check if this last state is the legitmate
        # -> NO! it just gets the last element of the output array

        qst_len_index_by_batch = tf.stack(
            [tf.range(batch_size,dtype=tf.int32),
             qst_len - 1], axis=1) #This yields the last index of each sequence ( -1 is
        # needed becuase sequence index starts with 0

        encoded_qst = tf.gather_nd(rnn_outputs, qst_len_index_by_batch)
        # tf.gather_nd, indices defines slices into the first N dimensions of params, where N = indices.shape[-1].

        self.encoded_qst = encoded_qst
        self.all_qst = rnn_outputs
        self.qst_len = qst_len

        encoded_img = build_conv(img, self.encoding_layers)
        encode_num_channels = self.encoding_layers[-1][0]
        reduced_height = int(height / (2 ** len(self.encoding_layers)))
        num_obj = reduced_height ** 2


        # self.get = [reduced_height, num_obj, encoded_img]

        encoded_img_flatten = tf.reshape(encoded_img, [batch_size, num_obj,
                                                       encode_num_channels])
        print(encoded_img_flatten.shape)

        # [b, d*d, # feature]

        encoded_qst_expand = tf.expand_dims(encoded_qst, axis=1) # [b, 1, # embed_dim]
        encoded_qst_tiled =  tf.tile(encoded_qst_expand, [1, num_obj, 1])

        print(encoded_qst_tiled.shape)

        encoded_img_qst = tf.concat([encoded_img_flatten, encoded_qst_tiled], axis=2)

        print(encoded_img_qst.shape)
        # [b, d*d, # feature + # qst encoding]

        encoded_img_qst = tf.transpose(encoded_img_qst, (0, 2, 1)) # for lower triangle
        # computation # [b, # feature + # qst encoding, d*d]
        encoded_img_qst_1 = tf.expand_dims(encoded_img_qst, axis = 3)
        encoded_img_qst_1 = tf.tile(encoded_img_qst_1, [1, 1, 1, num_obj])

        # self.encoded_img_qst_all = encoded_img_qst_1


        encoded_img_qst_1 = tf.matrix_band_part(encoded_img_qst_1, -1, 0) #lower triangle

        # self.encoded_img_qst_low = encoded_img_qst_1

        encoded_img_qst_2 = tf.expand_dims(encoded_img_qst, axis=2)
        encoded_img_qst_2 = tf.tile(encoded_img_qst_2, [1, 1, num_obj, 1])
        encoded_img_qst_2 = tf.matrix_band_part(encoded_img_qst_2, -1, 0)  # lower triangle

        encoded_img_qst_pair = tf.concat([encoded_img_qst_1, encoded_img_qst_2], axis=1)
        # [b, # channel, d*d, d*d]

        encoded_img_qst_pair = tf.transpose(encoded_img_qst_pair, [0, 2, 3, 1])
        # [b, d*d, d*d,  # channel]

        #TODO encoded_img_pst_pair includes self pairs (a_i, a_i) as well as (a_i, a_j)
        #TODO check if lower triangle operation is necessary for computational efficiency


        with tf.variable_scope('g_theta'):
            print('build g_theta')

            pair_output = build_mlp(encoded_img_qst_pair, self.g_theta_layers)
            mask = tf.reshape(tf.matrix_band_part(tf.ones([num_obj, num_obj]), -1,
                                                      0), [1, num_obj, num_obj, 1])
            pair_output_lower = tf.multiply(pair_output, mask)
            pair_output_sum = tf.reduce_sum(pair_output, (1, 2))

            # tf.assert_equal(pair_output_lower, pair_output)

        with tf.variable_scope('f_phi'):
            print('build f_phi')
            self.f_phi = build_mlp(pair_output_sum, self.f_phi_layers)

        with tf.variable_scope('output'):
            self.output = tf.layers.dense(self.f_phi, len(idx_to_value['answer']),
                                          use_bias=False) #use bias is false becuase it
            # this layer is a softmax activation layer

        with tf.variable_scope('loss'):
            # softmax = tf.nn.softmax(self.output)
            #
            # self.loss = -tf.reduce_mean(tf.one_hot(answer,
            #                                        depth=len(idx_to_value['answer'])) *
            #                           tf.log(tf.add(softmax, tf.constant(1e-12))))
            ans = tf.squeeze(ans, 1)

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=ans, logits=self.output))

        with tf.variable_scope('summary'):
            self.prediction = tf.argmax(self.output, axis=1)
            self.accuracy, _ = tf.metrics.accuracy(ans, self.prediction,
                                                 updates_collections=tf.GraphKeys.UPDATE_OPS)

            # self.average_loss, _ = tf.metrics.mean(self.loss,
            #                                     updates_collections=tf.GraphKeys.UPDATE_OPS)

            self.average_loss = self.loss

            # self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.prediction, ans),
            #                                     tf.float32))

            summary_trn = list()
            summary_trn.append(tf.summary.scalar('trn_accuracy', self.accuracy))
            summary_trn.append(tf.summary.scalar('trn_average_loss',
                                                      self.average_loss))

            self.summary_trn = tf.summary.merge(summary_trn)

            summary_test = list()
            summary_test.append(tf.summary.scalar('test_accuracy', self.accuracy))
            summary_test.append(tf.summary.scalar('test_average_loss',
                                                       self.average_loss))

            self.summary_test = tf.summary.merge(summary_test)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            # https://github.com/tensorflow/tensorflow/issues/19568 update_ops crashses
            # wehn rnn length is 32

            with tf.control_dependencies(self.update_ops):
                self.train_op = tf.train.AdamOptimizer().minimize(self.loss,
                                                               global_step=self.global_step)


