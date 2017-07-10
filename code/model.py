
import tensorflow as tf

class RelationalNetwork():

    def __init__(self, inputs, idx_to_value, embedding_size, g_theta_layers, f_phi_layers,
                 rnn_hidden_dim, is_train):


        self.g_theta_layers = g_theta_layers
        self.f_phi_layers = f_phi_layers
        self.rnn_hidden_dim = rnn_hidden_dim


        def build_mlp(inputs, layers):
            outputs = list()
            outputs.append(tf.contrib.layers.fully_connected(inputs, layers[0]))
            for layer_dim in layers[1:]:
                outputs.append(tf.contrib.layers.fully_connected(outputs[-1], layer_dim))
            return outputs[-1]

        def get_embedding_variable(var_name, embedding_size, inputs):
            vocab_size = len(idx_to_value[var_name])
            with tf.variable_scope('embedding_layer'):
                variable_embeddings = tf.get_variable(name='variable_embeddings_{}'.format(var_name),
                                                      shape=[vocab_size, embedding_size],
                                                      initializer=tf.random_uniform_initializer(-1, 1))

                embed_variable = tf.nn.embedding_lookup(variable_embeddings, inputs,
                                                        name='variable_lookup_{}'.format(var_name))

                if var_name in ['material', 'color', 'size', 'shape']:
                    shape = embed_variable.get_shape().as_list()
                    embed_variable = tf.reshape(embed_variable, (shape[0], -1, 2 * embedding_size))
            return embed_variable

        xyz_coords, pixel_coords, rotation, material, color, size, shape, question, \
        answer, num_pair, question_word_len = inputs

        max_num_pair = tf.reduce_max(num_pair)
        # num_pair = tf.cast(tf.reduce_mean(num_pair), tf.int32) #TODO 이렇게 하는게 맞나...

        material_embed = get_embedding_variable('material', embedding_size, material)
        color_embed = get_embedding_variable('color', embedding_size, color)
        size_embed = get_embedding_variable('size', embedding_size, size)
        shape_embed = get_embedding_variable('shape', embedding_size, shape)

        question_embed = get_embedding_variable('question', embedding_size, question)

        with tf.variable_scope('question_embedding'):
            rnn_cell = tf.contrib.rnn.LSTMCell(num_units=self.rnn_hidden_dim)
            outputs, last_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                     inputs=question_embed,
                                                     dtype=tf.float32
                                                     )
            # time_major = tf.transpose(outputs, (1,0,2))
            stacked = tf.stack([tf.range(question_word_len.get_shape()[0], dtype=tf.int64), question_word_len], axis=1)
            question_embedding = tf.gather_nd(outputs,stacked)
            self.question_embedding = question_embedding
            question_embedding = tf.expand_dims(question_embedding, axis=1)
            # question_embedding = outputs[:, -1, :]
        # self.question = question
        # self.stacked = stacked
        # self.rnn_hidden = outputs
        # self.question_word_len = question_word_len
        mask = tf.sequence_mask(num_pair, max_num_pair, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=2)
        question_embedding = tf.multiply(question_embedding, mask)

        # self.mask = mask
        # self.multi = question_embedding

        # 제대로 복사 되었는지 확인
        # assert all([all(question_embedding[0, i, :] == question_embedding [0,i+1,:]) for i in range(
        #     mask.get_shape()[1])])



        concat_vars = [xyz_coords, pixel_coords, material_embed, color_embed, size_embed, shape_embed, question_embedding]
        input_concat = tf.concat([xyz_coords, material_embed, color_embed, size_embed, shape_embed, question_embedding], axis=2)

        dimension_sum = int(6 + embedding_size * 2 * 4 + rnn_hidden_dim) # 6은 coords 변수 2개, embedding size에다가 obj1과 obj2 concat해서 *2하고 이런 변수가 5개니까 곱하기 5 + question embedding

        self.num_pair = num_pair
        with tf.variable_scope('g_theta'):
            input_concat = tf.reshape(input_concat, (-1, dimension_sum))
            obj_batch_output = build_mlp(input_concat, self.g_theta_layers)
            obj_pair_output = tf.reshape(obj_batch_output, (num_pair.get_shape().as_list()[0], -1, self.g_theta_layers[-1]))

            self.obj_pair_output = obj_pair_output
            self.mask = mask
            # obj_pair_output = tf.reshape(obj_batch_output, (-1, max_num_pair, self.g_theta_layers[-1])) # TODO 왜 max_num_pair을 하면 list of tensor가 들어왓다고 할가.

            masked_obj_pair_output = tf.multiply(obj_pair_output, mask)
            self.masked_obj_pair_output = masked_obj_pair_output

            # self.g_theta = tf.divide(tf.reduce_sum(masked_obj_pair_output, axis=1), tf.expand_dims(tf.cast(num_pair, tf.float32), axis=1)) / 1000
            self.g_theta = tf.reduce_mean(masked_obj_pair_output, axis=1)


        with tf.variable_scope('f_phi'):
            self.f_phi = build_mlp(self.g_theta, self.f_phi_layers)
        #
        with tf.variable_scope('output'):
            self.output = tf.contrib.layers.fully_connected(self.f_phi,
                                                            len(idx_to_value['answer']),
                                                            activation_fn=tf.nn.relu)

        with tf.variable_scope('loss'):
            softmax = tf.nn.softmax(self.output)

            self.loss = -tf.reduce_mean(tf.one_hot(answer,
                                                   depth=len(idx_to_value['answer'])) *
                                      tf.log(tf.add(softmax, tf.constant(1e-12))))

        with tf.variable_scope('accuracy'):
            self.prediction = tf.argmax(self.output, axis=1)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction, answer), tf.float32))

            # self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=self.output))

        if is_train:
            with tf.variable_scope('train'):
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss, global_step=self.global_step)

            # (outputs, output_state_fw, output_state_bw) = \
            #     tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            #         cells_fw=[rnn_cell],
            #         cells_bw=[rnn_cell],
            #         inputs=upg_embed,
            #         dtype=tf.float32)
            # eitem_no_feature = tf.concat([outputs[:, 0, :], outputs[:, -1, :]], axis=1)



