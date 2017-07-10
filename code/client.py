import model
import generate_data
import tensorflow as tf
import pickle
import numpy as np

def run_model(restore):
    len_key_list = [3, 4, 5, 6]
    generate_data.prepare_data(len_key_list)
    batch_size = 64
    # test_batch_size = 149991
    test_batch_size = 512
    num_epochs = 100
    # num_obj_pair = int(len_key * (len_key - 1) / 2)
    embedding_size = 4
    rnn_hidden_dim = 128
    g_theta_layers = [256] * 3
    f_phi_layers = [256, 512]

    with open('processed_data/idx_to_value.pkl', 'rb') as f:
        idx_to_value = pickle.load(f)

    with open('processed_data/question_answer_dict.pkl', 'rb') as f:
        word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word = pickle.load(f)

    idx_to_value['question'] = idx_to_word
    idx_to_value['answer'] = answer_idx_to_word

    for key in ['material', 'color', 'size', 'shape', 'question']:
        idx_to_value[key][0] = '_'
    # data_len_list = [len_key]

    with tf.Graph().as_default():
        with tf.name_scope('Train'):
            train_inputs = generate_data.inputs('train', len_key_list, batch_size, num_epochs, 3)
            with tf.variable_scope('Model', reuse=None):
                trn_model = model.RelationalNetwork(train_inputs, idx_to_value,
                                                    embedding_size, g_theta_layers,
                                                    f_phi_layers,
                                                    rnn_hidden_dim, True)

        with tf.name_scope('Validation'):
            test_inputs = generate_data.inputs('val', len_key_list, test_batch_size, num_epochs, 3)
            with tf.variable_scope('Model', reuse=True):
                test_model = model.RelationalNetwork(test_inputs, idx_to_value,
                                                     embedding_size, g_theta_layers,
                                                     f_phi_layers,
                                                     rnn_hidden_dim, False)

        count = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            if restore:
                saver.restore(sess, 'model/model.ckpt')
            try:
                while not coord.should_stop():

                    # stacked, qembed, rnn_hidden, question, q_len, mask, multi = sess.run([
                    #     trn_model.stacked,
                    #                         trn_model.question_embedding,
                    #               trn_model.rnn_hidden, trn_model.question,
                    #     trn_model.question_word_len,
                    # trn_model.mask,trn_model.multi])

                    # po, m, mpo = sess.run([trn_model.obj_pair_output, trn_model.mask,
                    #  trn_model.masked_obj_pair_output])
                    # print(a)
                    # print(a.shape)
                    _, global_step = sess.run([trn_model.train_op, trn_model.global_step])

                    if global_step % 1000 == 0:
                        test_loss, trn_loss, test_acc, trn_acc, trn_pred, test_pred = sess.run([test_model.loss, trn_model.loss, test_model.accuracy, trn_model.accuracy, trn_model.prediction, test_model.prediction])

                        print('global_step: {} | train loss : {} test loss : {} train acc : {} test acc : {}'.format(global_step, trn_loss, test_loss, trn_acc, test_acc))
                        print([idx_to_value['answer'][x] for x in trn_pred])
                        print([idx_to_value['answer'][x] for x in test_pred])
                        saver.save(sess, 'model/model.ckpt')
                    # a, b, c = sess.run([trn_model.g_theta, trn_model.f_phi, trn_model.output])
                    # print(a.shape)
                    # print(b.shape)
                    # print(c.shape)

                    count += 1

            except tf.errors.OutOfRangeError:
                print('Done training --epoch limit reached')
                print(count)
            finally:
                coord.request_stop()
            coord.join(threads)

        # sv = tf.train.Supervisor(logdir='ckpt_dir')
        # with sv.managed_session() as sess:
        #     sess.run(tf.local_variables_initializer())
        #     sess.run(tf.global_variables_initializer())
        #     coord = tf.train.Coordinator()
        #     threads = tf.train.start_queue_runners(sess, coord)
        #     try:
        #         while not coord.should_stop():
        #
        #             _, global_step = sess.run([trn_model.train_op, trn_model.global_step])
        #
        #             if global_step % 1000 == 0:
        #                 test_loss, trn_loss, test_acc, trn_acc, trn_pred, test_pred = sess.run([test_model.loss, trn_model.loss, test_model.accuracy, trn_model.accuracy, trn_model.prediction, test_model.prediction])
        #
        #                 print('global_step: {} | train loss : {} test loss : {} train acc : {} test acc : {}'.format(global_step, trn_loss, test_loss, trn_acc, test_acc))
        #                 print([idx_to_value['answer'][x] for x in trn_pred])
        #                 print([idx_to_value['answer'][x] for x in test_pred])

        #
        #             count += 1
        #
        #     except tf.errors.OutOfRangeError:
        #         print('Done training --epoch limit reached')
        #         print(count)
        #     finally:
        #         coord.request_stop()
        #     coord.join(threads)


if __name__ == '__main__':
    run_model(False)