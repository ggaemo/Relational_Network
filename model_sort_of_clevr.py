
import tensorflow as tf
import ops

class RelationalNetwork():

    def __init__(self, inputs, qst_color_vocab_size, qst_type_vocab_size, ans_vocab_size,
                 word_embedding_size, g_theta_layers, f_phi_layers, img_encoding_layers,
                 **kwargs):

        self.g_theta_layers = g_theta_layers
        self.f_phi_layers = f_phi_layers
        self.encoding_layers = img_encoding_layers
        self.is_training = tf.placeholder(tf.bool, shape=None)
        tf.add_to_collection('is_training', self.is_training)

        # self.qst_word = tf.placeholder(tf.string, shape=[None])
        # self.ans_word = tf.placeholder(tf.string, shape=[None])
        # self.pred_word = tf.placeholder(tf.string, shape=[None])
        self.img_pl = tf.placeholder(tf.float32, shape=[None, 75 + 20, 75, 3])




        # 20 is margin to put the question inside the image for summary view

        if 'batch_size' in kwargs:
            self.batch_size_for_learning_rate = kwargs['batch_size']

        if 'num_question' in kwargs:
            self.num_question = kwargs['num_question']

        if 'question_type_dict' in kwargs:
            self.question_type_dict = kwargs['question_type_dict']

        if 'base_learning_rate' in kwargs:
            self.base_learning_rate = kwargs['base_learning_rate']

        if 'cnn_reg' in kwargs:
            cnn_reg = kwargs['cnn_reg']

        if 'num_obj' in kwargs:
            num_obj = kwargs['num_obj']

        if 'reduced_height' in kwargs:
            reduced_height = kwargs['reduced_height']

        self.g_theta_activation = tf.placeholder(tf.float32,
                                                 shape=[None, num_obj, num_obj, 1])
        self.gate_pl = tf.placeholder(tf.float32,
                                      shape=[None, reduced_height, reduced_height, 1])

        def build_mlp(inputs, layers, drop_out=None):

            for layer_num, layer_dim in enumerate(layers):
                inputs = tf.layers.dense(inputs, layer_dim, activation=tf.nn.relu)
                if drop_out == layer_num:
                    inputs = tf.layers.dropout(inputs, rate=0.5,
                                                  training=self.is_training)
                    print('dropout')
                # bn_output = tf.layers.batch_normalization(fc_output,
                #                                          training =is_train)
                                                         # updates_collections=None) #decay
                # 0.99 or 0.95 or 0.90
                print(inputs.shape)
            return inputs

        def build_conv(input, layers):
            print('build convnet')
            for layer_num, layer_config in enumerate(layers):
                (num_filter, kernel_size, stride) = layer_config
                with tf.variable_scope('conv_layer_{}'.format(layer_num)):
                    input = ops.conv(input, num_filter, kernel_size, stride, cnn_reg,
                                     tf.nn.relu, self.is_training)
                    print(input.shape)
            return input


        def get_embedding_variable(inputs, vocab_size, embedding_size, name):
            # with tf.variable_scope('embedding_layer/{}'.format(name)):
            #     variable_embeddings = tf.get_variable(name='variable_embeddings',
            #                                           shape=[vocab_size, embedding_size],
            #                                           initializer=tf.random_uniform_initializer(-1, 1))
            #
            #     embed_variable = tf.nn.embedding_lookup(variable_embeddings, inputs,
            #                                             name='variable_lookup')

            with tf.variable_scope('onehot/{}'.format(name)):
                embed_variable = tf.one_hot(inputs, vocab_size)
            return embed_variable

        def build_coord_tensor(batch_size, height):
            # coord = tf.linspace(0.0, height - 1, height)
            coord = tf.linspace(-height/2, height/2, height)
            x = tf.tile(tf.expand_dims(coord, 0), [height, 1])
            y = tf.tile(tf.expand_dims(coord, 1), [1, height])

            coord_xy = tf.stack((x, y), axis=2)
            coord_xy_batch = tf.tile(tf.expand_dims(coord_xy, 0), [batch_size, 1, 1, 1])
            coord_xy_batch = tf.cast(coord_xy_batch, tf.float32)
            print('coord_xy shape', coord_xy_batch.shape)
            return coord_xy_batch



        img = inputs['img']
        ans = inputs['ans']
        qst_color = inputs['qst_c']
        qst_type = inputs['qst_type']
        qst_subtype = inputs['qst_subtype']

        self.qst_color = qst_color
        self.qst_type = qst_type
        self.qst_subtype = qst_subtype



        qst_subtype_len = 3

        qst_type = qst_type * qst_subtype_len  + qst_subtype

        self.img = img
        self.ans = ans
        self.qst = tf.concat([tf.expand_dims(qst_color,1), tf.expand_dims(qst_type, 1)],
                             axis=1)


        tf.add_to_collection('img', img)
        tf.add_to_collection('ans', ans)
        tf.add_to_collection('qst', self.qst)
        tf.add_to_collection('qst_color', qst_color)
        tf.add_to_collection('qst_type', qst_type)


        # qst_color, qst_type = tf.split(qst, 2, axis=1)


        _, height, width, num_input_channel = img.get_shape().as_list()
        batch_size = tf.shape(img)[0]

        # do this if set_shape is not done

        # height = tf.shape(img)[1]
        # width = tf.shape(img)[2]
        # num_input_channel = tf.shape(img)[-1]

        with tf.variable_scope('question_embedding'):
            qst_color_embed = get_embedding_variable(qst_color, qst_color_vocab_size,
                                                     word_embedding_size, 'question_color')
            qst_type_embed = get_embedding_variable(qst_type, qst_type_vocab_size,
                                                     word_embedding_size, 'question_type')

            encoded_qst = tf.concat([qst_color_embed, qst_type_embed], axis=1)

            tf.add_to_collection('encoded_qst', encoded_qst)


        with tf.variable_scope('image_embedding'):
            encoded_img = build_conv(img, self.encoding_layers)


            reduced_height = tf.cast(tf.ceil(height / (2 ** len(self.encoding_layers))),
                                     tf.int32)
            num_obj = reduced_height ** 2

            coord_tensor = build_coord_tensor(batch_size, reduced_height)

            encoded_img_coord = tf.concat([encoded_img, coord_tensor], axis=3)


        with tf.variable_scope('image_object_pairing'):
            print('encoded img_coord', encoded_img_coord.shape)

            encode_num_channels = self.encoding_layers[-1][0]
            encoded_img_flatten = tf.reshape(encoded_img_coord, [batch_size, num_obj,
                                                                 encode_num_channels + 2])

            #coord num channel 2

            print('encoded flatten', encoded_img_flatten.shape)
            # [b, d*d, # feature]



            encoded_img_flatten = tf.transpose(encoded_img_flatten, (0, 2, 1)) # for lower triangle
            # computation # [b, # feature , d*d]
            encoded_img_flatten_1 = tf.expand_dims(encoded_img_flatten, axis = 3)
            encoded_img_flatten_1 = tf.tile(encoded_img_flatten_1, [1, 1, 1, num_obj])


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
                                            [batch_size, qst_type_vocab_size + qst_color_vocab_size, 1, 1])



            encoded_qst_tiled = tf.tile(encoded_qst_expand, [1, 1, num_obj, num_obj])
            encoded_qst_tiled = tf.matrix_band_part(encoded_qst_tiled, -1,
                                                    0, name='lower_matrix')  # lower
            # triangle

            print('encoded tiled', encoded_qst_tiled.shape)

            encoded_img_qst_pair = tf.concat([encoded_img_pair, encoded_qst_tiled], axis=1)

            # [b, # channel + #rnn dim, d*d, d*d]

        encoded_img_qst_pair = tf.transpose(encoded_img_qst_pair, [0, 2, 3, 1])

        # [b, d*d, d*d,  # channel + # rnn dim]

        #TODO encoded_img_pst_pair includes self pairs (a_i, a_i) as well as (a_i, a_j)
        #TODO check if lower triangle operation is necessary for computational efficiency


        with tf.variable_scope('g_theta'):
            print('build g_theta')

            pair_output = build_mlp(encoded_img_qst_pair, self.g_theta_layers)

            mask = tf.linalg.LinearOperatorLowerTriangular(
                tf.ones([num_obj, num_obj]))

            mask = tf.reshape(mask.to_dense(), [1, num_obj, num_obj, 1]) # lower triangle

            pair_output_lower = tf.multiply(pair_output, mask)
            self.pair_output_lower = pair_output_lower
            self.pair_output_lower_activation = tf.reduce_sum(tf.abs(pair_output_lower),
                                                              3, keep_dims=True)

            tf.add_to_collection('g_theta', pair_output_lower)

            # pair_output_lower = pair_output
            pair_output_sum = tf.reduce_sum(pair_output_lower, (1, 2))


        with tf.variable_scope('f_phi'):
            print('build f_phi')
            self.f_phi = build_mlp(pair_output_sum, self.f_phi_layers)
            print('no drop out')

        with tf.variable_scope('output'):
            self.output = tf.layers.dense(self.f_phi, ans_vocab_size,
                                          use_bias=False) #use bias is false becuase it
            # this layer is a softmax activation layer

        with tf.variable_scope('loss'):
            xent_loss_raw =tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=ans, logits=self.output)
            xent_loss_raw = tf.check_numerics(xent_loss_raw, 'nan value found in loss raw')
            self.xent_loss = tf.reduce_mean(xent_loss_raw)
            self.loss = self.xent_loss


        with tf.variable_scope('learning_rate'):
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.epoch = tf.Variable(0, trainable=False, name='epoch')
            self.increment_epoch_op = tf.assign(self.epoch, self.epoch + 1)

            if self.batch_size_for_learning_rate < 64:
                self.learning_rate = self.base_learning_rate
            else:
                self.learning_rate = tf.train.polynomial_decay(self.base_learning_rate,
                                                      self.epoch,
                                                      decay_steps=5,
                                                      end_learning_rate=self.base_learning_rate *(self.batch_size_for_learning_rate/64),
                                                      )

        with tf.variable_scope('summary'):

            self.prediction = tf.argmax(self.output, axis=1)

            tf.add_to_collection('pred', self.prediction)

            self.accuracy, _ = tf.metrics.accuracy(ans, self.prediction,
                                                 updates_collections='summary_update')

            summary_trn = list()
            summary_trn.append(tf.summary.scalar('trn_accuracy', self.accuracy))
            summary_trn.append(tf.summary.scalar('learning_rate', self.learning_rate))

            summary_test = list()
            summary_test.append(tf.summary.scalar('test_accuracy', self.accuracy))

            for key, val in self.question_type_dict.items():
                acc_type_mask = tf.equal(qst_type, key)
                ans_tmp = tf.boolean_mask(ans, acc_type_mask)
                pred_tmp = tf.boolean_mask(self.prediction, acc_type_mask)
                acc_tmp, _ = tf.metrics.accuracy(ans_tmp, pred_tmp,
                                     updates_collections='summary_update')
                summary_trn.append(tf.summary.scalar('trn_{}_acc'.format(val), acc_tmp))
                summary_test.append(tf.summary.scalar('test_{}_acc'.format(val), acc_tmp))

            nonrelational_mask = tf.less_equal(qst_type, 2)
            relational_mask = tf.greater_equal(qst_type, 3)

            for rel_nonrel, mask in zip(['nonrel', 'rel'], [nonrelational_mask,
                                                       relational_mask]):
                ans_tmp = tf.boolean_mask(ans, mask)
                pred_tmp = tf.boolean_mask(self.prediction, mask)
                acc_tmp, _ = tf.metrics.accuracy(ans_tmp, pred_tmp,
                                                 updates_collections='summary_update')
                summary_trn.append(tf.summary.scalar('trn_{}_acc'.format(rel_nonrel), acc_tmp))
                summary_test.append(tf.summary.scalar('test_{}_acc'.format(rel_nonrel),
                                                      acc_tmp))

            self.summary_trn = tf.summary.merge(summary_trn)

            self.summary_test = tf.summary.merge(summary_test)

            '''
            summaries that consumes batches
            '''
            trn_loss_summary = [tf.summary.scalar('trn_xent_loss', self.xent_loss)]

            test_loss_summary = [tf.summary.scalar('test_xent_loss', self.xent_loss)]

            self.trn_loss_summary = tf.summary.merge(trn_loss_summary)

            self.test_loss_summary = tf.summary.merge(test_loss_summary)

        with tf.variable_scope('img_qst_summary'):
            additional = list()
            additional.append(tf.summary.image('img', self.img_pl, max_outputs=10))
            additional.append(tf.summary.image('g_theta_output', self.g_theta_activation,
                                               max_outputs=10))
            additional.append(tf.summary.image('gate', self.gate_pl,
                                               max_outputs=10))
            # additional.append(tf.summary.text('ans', self.ans_word))
            # additional.append(tf.summary.text('question', self.qst_word))
            # additional.append(tf.summary.text('prediction', self.pred_word))
            self.summary_additional = tf.summary.merge(additional)

        with tf.variable_scope('train'):

            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            self.summary_update_ops = tf.get_collection('summary_update')
            self.assert_ops = tf.get_collection('assert')

            with tf.control_dependencies(self.update_ops + self.assert_ops + self.summary_update_ops):
                self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(
                    self.loss,global_step=self.global_step)