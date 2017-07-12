import model
import generate_data
import tensorflow as tf
import pickle
import time

def run_model(restore):
    generate_data.prepare_data()
    batch_size = 64
    test_data_size = 149991
    test_batch_size = 8823 #second biggest
    num_epochs = 10000
    cat_embedding_size = 4
    word_embedding_size = 32
    rnn_hidden_dim = 128
    g_theta_layers = [512, 512, 512, 512]
    f_phi_layers = [512, 1024]


    with open('model_config.txt', 'w') as f:
        f.write('batch size : {}'.format(batch_size))
        f.write('num epoch : {}'.format(num_epochs))
        f.write('cat_embedding size: {}'.format(cat_embedding_size))
        f.write('word_embedding size: {}'.format(word_embedding_size))
        f.write('rnn_hidden_dim: {}'.format(rnn_hidden_dim))
        f.write('g_theta layers: {}'.format(g_theta_layers))
        f.write('f_phi_layers: {}'.format(f_phi_layers))

    with open('processed_data/idx_to_value.pkl', 'rb') as f:
        idx_to_value = pickle.load(f)

    with open('processed_data/question_answer_dict.pkl', 'rb') as f:
        word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word = pickle.load(f)

    idx_to_value['question'] = idx_to_word
    idx_to_value['answer'] = answer_idx_to_word

    for key in ['material', 'color', 'size', 'shape', 'question']:
        idx_to_value[key][0] = '_'

    with tf.Graph().as_default():
        with tf.name_scope('Train'):
            train_inputs = generate_data.inputs('train', batch_size, num_epochs, 3)
            with tf.variable_scope('Model', reuse=None):
                trn_model = model.RelationalNetwork(train_inputs, idx_to_value,
                                                    cat_embedding_size,
                                                    word_embedding_size, g_theta_layers,
                                                    f_phi_layers,
                                                    rnn_hidden_dim, True)

        with tf.name_scope('Validation'):
            test_inputs = generate_data.inputs('val', test_batch_size, num_epochs, 3)
            with tf.variable_scope('Model', reuse=True):
                test_model = model.RelationalNetwork(test_inputs, idx_to_value,
                                                     cat_embedding_size,
                                                     word_embedding_size, g_theta_layers,
                                                     f_phi_layers,
                                                     rnn_hidden_dim, False)


        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            if restore:
                latest_model = tf.train.latest_checkpoint('model')
                print('restored model from ', latest_model)
                saver.restore(sess, latest_model)
            try:
                prev = time.time()
                while not coord.should_stop():

                    _, global_step = sess.run([trn_model.train_op, trn_model.global_step])

                    if global_step % 5000 == 0:
                        now = time.time()
                        test_loss = 0
                        test_acc = 0
                        for _ in range(test_data_size // test_batch_size + 1):
                            test_loss_batch, test_acc_batch = sess.run([test_model.loss,
                                                             test_model.accuracy])
                            test_loss += test_loss_batch
                            test_acc += test_acc_batch

                        test_loss = test_loss / test_data_size * test_batch_size
                        test_acc = test_acc / test_data_size * test_batch_size

                        print('global_step: {} | test loss : {} test acc : {}'.format(global_step, test_loss, test_acc))
                        minutes = (now - prev) / 60
                        print('took {} min, running at {} samples / min'.format(minutes,
                            batch_size * 5000 / minutes))

                        prev = time.time()

                        # print([idx_to_value['answer'][x] for x in trn_pred])
                        # print([idx_to_value['answer'][x] for x in test_pred])
                        saver.save(sess, 'model/model.ckpt', global_step=global_step)

            except tf.errors.OutOfRangeError:
                print('Done training --epoch limit reached')
            finally:
                coord.request_stop()
            coord.join(threads)

if __name__ == '__main__':
    run_model(True)
