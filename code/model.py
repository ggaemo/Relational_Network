
import tensorflow as tf

class RelationalNetwork():

    def __init__(self, inputs, idx_to_value, cat_embedding_size,
                 word_embedding_size, g_theta_layers, f_phi_layers,
                 rnn_hidden_dim, is_train):

        self.g_theta_layers = g_theta_layers
        self.f_phi_layers = f_phi_layers
        self.rnn_hidden_dim = rnn_hidden_dim


        def build_mlp(inputs, layers):
            outputs = list()
            outputs.append(inputs)
            # outputs.append(tf.contrib.layers.fully_connected(inputs, layers[0]))
            for layer_dim in layers:
                layer_input = outputs[-1]
                fc_output = tf.layers.dense(layer_input, layer_dim, activation=tf.nn.relu)
                # bn_output = tf.layers.batch_normalization(fc_output,
                #                                          training =is_train)
                                                         # updates_collections=None) #decay
                # 0.99 or 0.95 or 0.90
                outputs.append(fc_output)
            return outputs[-1]

        def get_embedding_variable(var_name, embedding_size, inputs):
            vocab_size = len(idx_to_value[var_name])
            with tf.variable_scope('embedding_layer'):
                variable_embeddings = tf.get_variable(name='variable_embeddings_{}'.format(var_name),
                                                      shape=[vocab_size, embedding_size],
                                                      initializer=tf.random_uniform_initializer(-1, 1))

                embed_variable = tf.nn.embedding_lookup(variable_embeddings, inputs,
                                                        name='variable_lookup_{}'.format(var_name))

                # embed_variable.shape (batch_size, max_num_pair, 2, embedding_var)

                if var_name in ['material', 'color', 'size', 'shape']:
                    shape = embed_variable.get_shape().as_list()
                    embed_variable = tf.reshape(embed_variable, (shape[0], -1, 2 * embedding_size))
                    # change to shape (batch_size, max_num_pair, 2 * embedding_var)
            return embed_variable

        xyz_coords, pixel_coords, rotation, material, color, size, shape, question, \
        answer, num_pair, question_word_len = inputs

        max_num_pair = tf.reduce_max(num_pair)
        # num_pair = tf.cast(tf.reduce_mean(num_pair), tf.int32) #TODO 이렇게 하는게 맞나...

        material_embed = get_embedding_variable('material', cat_embedding_size, material)
        color_embed = get_embedding_variable('color', cat_embedding_size, color)
        size_embed = get_embedding_variable('size', cat_embedding_size, size)
        shape_embed = get_embedding_variable('shape', cat_embedding_size, shape)

        question_embed = get_embedding_variable('question', word_embedding_size, question)

        with tf.variable_scope('question_embedding'):
            rnn_cell = tf.contrib.rnn.LSTMCell(num_units=self.rnn_hidden_dim)
            outputs, last_states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                     inputs=question_embed,
                                                     dtype=tf.float32
                                                     )

            question_word_len_index_by_batch = tf.stack(
                [tf.range(question_word_len.get_shape()[0],dtype=tf.int64),
                 question_word_len], axis=1)
            # for each instance, get the last hidden vector (this is needed because the
            # length of each question is different and padding of 0 occurs with batching

            question_embedding = tf.gather_nd(outputs,question_word_len_index_by_batch)
            # tf.gather_nd, indices defines slices into the first N dimensions of params, where N = indices.shape[-1].
            question_embedding = tf.expand_dims(question_embedding, axis=1) # to make
            # this as a sequence of length 1.
            # question_embedding.shape = (batch_size, 1, question_word_embedding_size)

        mask = tf.sequence_mask(num_pair, max_num_pair, dtype=tf.float32)
        mask = tf.expand_dims(mask, axis=2)
        question_embedding = tf.multiply(question_embedding, mask) # supports broadcasting

        # question_embedding.shape (batch_size, max_num_pair, question_word_embedding_size)

        # concat_vars = [xyz_coords, pixel_coords, material_embed, color_embed, size_embed, shape_embed, question_embedding]
        # pixel coords is polar coordinates so it has the same information as xyz_coords
        input_concat = tf.concat([xyz_coords, material_embed, color_embed, size_embed, shape_embed, question_embedding], axis=2)

        dimension_sum = int(6 + cat_embedding_size * 2 * 4 + rnn_hidden_dim) # 6은 coords 변수
        #  2개, embedding size에다가 obj1과 obj2 concat해서 *2하고 이런 변수가 4개니까 곱하기 4 + question
        # embedding

        # self.num_pair = num_pair

        with tf.variable_scope('g_theta'):
            batch_size = num_pair.get_shape().as_list()[0]
            input_concat.set_shape((batch_size, None, dimension_sum))
            # ValueError: The last dimension of the inputs to `Dense` should be defined. Found `None`.

            obj_pair_output = build_mlp(input_concat, self.g_theta_layers)
            # obj_batch_output = build_mlp(input_concat, self.g_theta_layers)
            # obj_pair_output = tf.reshape(obj_batch_output, (batch_size, -1,
            #                                                 self.g_theta_layers[-1]))

            masked_obj_pair_output = tf.multiply(obj_pair_output, mask)

            self.mask_obj_pair = masked_obj_pair_output
            self.obj_pair = obj_pair_output
            # attention_layer = 1
            # attention_by_instance = tf.contrib.layers.fully_connected(input_concat,
            #                                                   attention_layer)
            #
            # attention = tf.expand_dims(tf.reshape(attention_by_instance,
            #                                        (batch_size, -1)), axis=2)

            # self.g_theta = tf.reduce_sum(tf.multiply(masked_obj_pair_output,
            #                                          attention), axis=1)

            # self.g_theta = tf.reduce_sum(masked_obj_pair_output, axis=1)

            self.g_theta = tf.divide(tf.reduce_sum(masked_obj_pair_output, axis=1),
            tf.expand_dims(tf.cast(num_pair, tf.float32), axis=1)) #TODO sum이 중요하다.

            # mean보다는 sum이 훨씬 작동 잘한다. absolute value가 역할을 한다.

        with tf.variable_scope('f_phi'):
            self.f_phi = build_mlp(self.g_theta, self.f_phi_layers)
        #
        with tf.variable_scope('output'):
            self.output = tf.layers.dense(self.f_phi, len(idx_to_value['answer']))

        with tf.variable_scope('loss'):
            # softmax = tf.nn.softmax(self.output)
            #
            # self.loss = -tf.reduce_mean(tf.one_hot(answer,
            #                                        depth=len(idx_to_value['answer'])) *
            #                           tf.log(tf.add(softmax, tf.constant(1e-12))))

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=answer, logits=self.output))

        with tf.variable_scope('accuracy'):
            self.prediction = tf.argmax(self.output, axis=1)
            self.accuracy = tf.reduce_sum(tf.cast(tf.equal(self.prediction, answer),
                                                tf.float32))

        if is_train:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.variable_scope('train'):
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                with tf.control_dependencies(update_ops):
                    self.train_op = tf.train.AdamOptimizer().minimize(self.loss,
                                                                   global_step=self.global_step)
