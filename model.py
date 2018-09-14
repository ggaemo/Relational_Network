
import tensorflow as tf
import ops

class RelationalNetwork():

    def __init__(self, inputs, qst_vocab_size, ans_vocab_size,
                 word_embedding_size, g_theta_layers, f_phi_layers, img_encoding_layers,
                 rnn_hidden_dim):

        self.g_theta_layers = g_theta_layers
        self.f_phi_layers = f_phi_layers
        self.encoding_layers = img_encoding_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.is_training = tf.placeholder(tf.bool, shape=None)



        def build_mlp(inputs, layers, drop_out=None):
            print('build mlp')
            outputs = list()
            outputs.append(inputs)

            for layer_num, layer_dim in enumerate(layers):
                layer_input = outputs[-1]
                fc_output = tf.layers.dense(layer_input, layer_dim, activation=tf.nn.relu)
                if drop_out == layer_num:
                    fc_output = tf.layers.dropout(fc_output, rate=0.5,
                                                  training=self.is_training)
                    print('dropout')
                # bn_output = tf.layers.batch_normalization(fc_output,
                #                                          training =is_train)
                                                         # updates_collections=None) #decay
                # 0.99 or 0.95 or 0.90
                print(fc_output.shape)
                outputs.append(fc_output)
            return outputs[-1]

        def build_conv(input, layers):
            print('build convnet')
            for layer_num, layer_config in enumerate(layers):
                (num_filter, kernel_size, stride) = layer_config
                with tf.variable_scope('conv_layer_{}'.format(layer_num)):
                    input = ops.conv(input, num_filter, kernel_size, stride, 'bn',
                                     tf.nn.relu, self.is_training)
                    print(input.shape)
            return input

        def build_conv_transpose(input, layers):
            print('build conv transpose net')
            for layer_num, layer_config in enumerate(layers[:-1]):
                (num_filter, kernel_size, stride) = layer_config
                with tf.variable_scope('conv_layer_{}'.format(layer_num)):
                    input = ops.conv_transpose(input, num_filter, kernel_size, stride,
                                               'bn',
                                     tf.nn.relu, self.is_training)
                    print(input.shape)

            (num_filter, kernel_size, stride) = layers[-1]
            with tf.variable_scope('conv_layer_{}'.format(layer_num+1)):
                input = ops.conv_transpose(input, num_filter, kernel_size, stride,
                                           'bn',
                                           tf.nn.tanh, self.is_training)
            return input

        def get_embedding_variable(inputs, vocab_size, embedding_size):
            with tf.variable_scope('embedding_layer'):
                variable_embeddings = tf.get_variable(name='variable_embeddings',
                                                      shape=[vocab_size, embedding_size],
                                                      initializer=tf.random_uniform_initializer(-1, 1))

                embed_variable = tf.nn.embedding_lookup(variable_embeddings, inputs,
                                                        name='variable_lookup')
            return embed_variable

        def build_coord_tensor(batch_size, height):
            coord = tf.linspace(-height / 2, height / 2, height)
            x = tf.tile(tf.expand_dims(coord, 0), [height, 1])
            y = tf.tile(tf.expand_dims(coord, 1), [1, height])

            coord_xy = tf.stack((x, y), axis=2)
            coord_xy_batch = tf.tile(tf.expand_dims(coord_xy, 0), [batch_size, 1, 1, 1])
            print('coord_xy shape', coord_xy_batch.shape)
            return coord_xy_batch

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

        with tf.variable_scope('question_embedding'):
            question_embed = get_embedding_variable(qst, qst_vocab_size,
                                                    word_embedding_size)
            rnn_cell = tf.contrib.rnn.GRUCell(num_units=self.rnn_hidden_dim)
            rnn_outputs, last_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                         inputs=question_embed,
                                                         dtype=tf.float32,
                                                         sequence_length=qst_len,
                                                         parallel_iterations=71
                                                     )
            # GRU
            encoded_qst = last_states
            #LSTM
            # c, h = last_states
            # encoded_qst = h

            #if parallel_iteration is given 71, it overcomes some strange error
            # tensorflow dynamic rnn puts out when the length is 32

            '''
            if you want to use rnn_outputs, use the below  
            '''
            # qst_len_index_by_batch = tf.stack(
            #     [tf.range(batch_size,dtype=tf.int32),
            #      qst_len - 1], axis=1) #This yields the last index of each sequence ( -1 is
            # # needed becuase sequence index starts with 0
            #
            # encoded_qst = tf.gather_nd(rnn_outputs, qst_len_index_by_batch)
            # # tf.gather_nd, indices defines slices into the first N dimensions of params, where N = indices.shape[-1].

        with tf.variable_scope('image_embedding'):
            encoded_img = build_conv(img, self.encoding_layers)


            encode_num_channels = self.encoding_layers[-1][0]
            reduced_height = int(height / (2 ** len(self.encoding_layers)))
            num_obj = reduced_height ** 2

            coord_tensor = build_coord_tensor(batch_size, reduced_height)

            encoded_img_coord = tf.concat([encoded_img, coord_tensor], axis=3)

            # self.get = [coord_tensor, encoded_img_coord]

        with tf.variable_scope('decoder'):
            self.decoding_layers = self.encoding_layers
            self.decoding_layers[-1][0] = 3 # last channel to have 3 channels
            recon = build_conv_transpose(encoded_img, self.decoding_layers)



        with tf.variable_scope('image_object_pairing'):
            print('encoded img_coord', encoded_img_coord.shape)
            encoded_img_flatten = tf.reshape(encoded_img_coord, [batch_size, num_obj,
                                                           encode_num_channels + 2])
            #coord num channel 2

            print(encoded_img_flatten.shape)
            # [b, d*d, # feature]

            # encoded_img_qst = tf.concat([encoded_img_flatten, encoded_qst_tiled], axis=2)

            encoded_img_flatten = tf.transpose(encoded_img_flatten, (0, 2, 1)) # for lower triangle
            # computation # [b, # feature , d*d]
            encoded_img_flatten_1 = tf.expand_dims(encoded_img_flatten, axis = 3)
            encoded_img_flatten_1 = tf.tile(encoded_img_flatten_1, [1, 1, 1, num_obj])

            # self.encoded_img_qst_all = encoded_img_qst_1

            encoded_img_flatten_1 = tf.matrix_band_part(encoded_img_flatten_1, -1, 0) #lower#
            #  triangle

            encoded_img_flatten_2 = tf.expand_dims(encoded_img_flatten, axis=2)
            encoded_img_flatten_2 = tf.tile(encoded_img_flatten_2, [1, 1, num_obj, 1])
            encoded_img_flatten_2 = tf.matrix_band_part(encoded_img_flatten_2, -1, 0)  # lower triangle

            encoded_img_pair = tf.concat([encoded_img_flatten_1, encoded_img_flatten_2],
                                             axis=1)
            # [b, # channel, d*d, d*d]

        with tf.variable_scope('img_qst_concat'):
            encoded_qst_expand = tf.reshape(encoded_qst,
                                            [batch_size, rnn_hidden_dim, 1, 1])
            # [b, 1,
            #  rnn_hidden_dim]
            encoded_qst_tiled = tf.tile(encoded_qst_expand, [1, 1, num_obj, num_obj])
            encoded_qst_tiled = tf.matrix_band_part(encoded_qst_tiled, -1,
                                                    0)  # lower triangle

            print(encoded_qst_tiled.shape)

            encoded_img_qst_pair = tf.concat([encoded_img_pair, encoded_qst_tiled], axis=1)

            # [b, # channel + #rnn dim, d*d, d*d]

        tf.add_to_collection('assert', tf.assert_equal(encoded_img_qst_pair, tf.matrix_band_part(
            encoded_img_qst_pair,-1, 0), message='qst lower'))

        encoded_img_qst_pair = tf.transpose(encoded_img_qst_pair, [0, 2, 3, 1])
        # [b, d*d, d*d,  # channel + # rnn dim]

        #TODO encoded_img_pst_pair includes self pairs (a_i, a_i) as well as (a_i, a_j)
        #TODO check if lower triangle operation is necessary for computational efficiency


        with tf.variable_scope('g_theta'):
            print('build g_theta')

            pair_output = build_mlp(encoded_img_qst_pair, self.g_theta_layers)
            mask = tf.reshape(tf.matrix_band_part(tf.ones([num_obj, num_obj]), -1,
                                                      0), [1, num_obj, num_obj, 1])
            pair_output_lower = tf.multiply(pair_output, mask)
            pair_output_sum = tf.reduce_sum(pair_output_lower, (1, 2))

            # self.a = tf.assert_equal(pair_output_lower, pair_output,
            #                                                message='lower_pair')
            #
            # self.get = [pair_output_lower, pair_output]

        with tf.variable_scope('f_phi'):
            print('build f_phi')
            self.f_phi = build_mlp(pair_output_sum, self.f_phi_layers,
                                   len(self.f_phi_layers) - 1)

            print('dropout at last layer')

        with tf.variable_scope('output'):
            self.output = tf.layers.dense(self.f_phi, ans_vocab_size,
                                          use_bias=False) #use bias is false becuase it
            # this layer is a softmax activation layer

        with tf.variable_scope('loss'):

            ans = tf.squeeze(ans, 1)
            xent_loss_raw =tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=ans, logits=self.output)
            xent_loss_raw = tf.check_numerics(xent_loss_raw, 'nan value found '
                                                                       'in '
                                                                  'loss raw')
            self.xent_loss = tf.reduce_mean(xent_loss_raw)

            self.recon_loss = tf.losses.absolute_difference(img, recon)

            self.loss = self.xent_loss + self.recon_loss

        with tf.variable_scope('summary'):
            self.prediction = tf.argmax(self.output, axis=1)
            self.accuracy, _ = tf.metrics.accuracy(ans, self.prediction,
                                                 updates_collections=tf.GraphKeys.UPDATE_OPS)

            # self.average_loss, _ = tf.metrics.mean(self.loss,
            #                                     updates_collections=tf.GraphKeys.UPDATE_OPS)

            # self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.prediction, ans),
            #                                     tf.float32))

            summary_trn = list()
            summary_trn.append(tf.summary.scalar('trn_accuracy', self.accuracy))

            trn_loss_summary = [tf.summary.scalar('trn_recon_loss', self.recon_loss),
                                     tf.summary.scalar('trn_xent_loss', self.xent_loss)]

            self.trn_loss_summary = tf.summary.merge(trn_loss_summary)

            self.summary_trn = tf.summary.merge(summary_trn)

            summary_test = list()
            summary_test.append(tf.summary.scalar('test_accuracy', self.accuracy))

            test_loss_summary = [tf.summary.scalar('test_recon_loss', self.recon_loss),
                                tf.summary.scalar('test_xent_loss', self.xent_loss)]

            self.test_loss_summary = tf.summary.merge(test_loss_summary)

            self.summary_test = tf.summary.merge(summary_test)

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.assert_ops = tf.get_collection('assert')

        with tf.variable_scope('train'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')

            # https://github.com/tensorflow/tensorflow/issues/19568 update_ops crashses
            # wehn rnn length is 32

            with tf.control_dependencies(self.update_ops + self.assert_ops):
                self.train_op = tf.train.AdamOptimizer(2.5*10e-4).minimize(self.loss,
                                                               global_step=self.global_step)


