import model
import generate_data
import tensorflow as tf
import pickle
import time
import argparse
import os
import re

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int)
parser.add_argument('-test_data_size', type=int, default=149991)
parser.add_argument('-test_batch_size', type=int, default=128, help='second biggest '
                                                                    'denominator')
parser.add_argument('-num_epochs', type=int, default=10000)
parser.add_argument('-cat_embedding_size', type=int)
parser.add_argument('-word_embedding_size', type=int)
parser.add_argument('-rnn_hidden_dim', type=int)
parser.add_argument('-g_theta_layers', type=int, nargs='+')
parser.add_argument('-f_phi_layers', type=int, nargs='+')
parser.add_argument('-option', type=str)
parser.add_argument('-restore', action='store_true', default=False)
args = parser.parse_args()

generate_data.prepare_data()
batch_size = args.batch_size
test_data_size = args.test_data_size
test_batch_size = args.test_batch_size
num_epochs = args.num_epochs
cat_embedding_size = args.cat_embedding_size
word_embedding_size = args.word_embedding_size
rnn_hidden_dim = args.rnn_hidden_dim
g_theta_layers = args.g_theta_layers # [512, 512, 512]
f_phi_layers = args.f_phi_layers # [512, 1024]
option = args.option
restore = args.restore

with open('model_config.txt', 'w') as f:
    f.write('batch size : {}'.format(batch_size))
    f.write('num epoch : {}'.format(num_epochs))
    f.write('cat_embedding size: {}'.format(cat_embedding_size))
    f.write('word_embedding size: {}'.format(word_embedding_size))
    f.write('rnn_hidden_dim: {}'.format(rnn_hidden_dim))
    f.write('g_theta layers: {}'.format(g_theta_layers))
    f.write('f_phi_layers: {}'.format(f_phi_layers))

model_dir = 'model_{}_{}_{}_{}_{}_{}/'.format(cat_embedding_size,
                                              word_embedding_size,
                                              rnn_hidden_dim,
                                              '-'.join([str(x) for x in g_theta_layers]),
                                              '-'.join([str(x) for x in f_phi_layers]),
                                              option)
model_dir = model_dir+'_model_specifics'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
elif not restore:
    print('directory exists')
    raise FileExistsError

terminal_output = open(model_dir+'terminal_output.txt', 'w')


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
    config = tf.ConfigProto()
    config.gpu_options.allow_growth =True
    with tf.Session(config=config) as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        if restore:
            latest_model = tf.train.latest_checkpoint(model_dir)
            a = model_dir.split('/')[0]

            print('restored model from ', latest_model)
            saver.restore(sess, latest_model)
        try:
            prev = time.time()
            while True:

                # num_pair, output, g_theta, g_theta_mean = sess.run([
                #     trn_model.num_pair, trn_model.masked_obj_pair_output,
                #             trn_model.g_theta, trn_model.g_theta_mean])
                # ai, a, g = sess.run([trn_model.attention_by_instance,
                #                     trn_model.attention,
                #                     trn_model.g_theta])

                _, global_step = sess.run([trn_model.train_op, trn_model.global_step])

                if global_step % 500 == 0:

                    trn_loss, trn_acc = sess.run([trn_model.loss,
                                                     trn_model.accuracy])
                    trn_acc = trn_acc/ batch_size
                    now = time.time()
                    test_loss = 0
                    test_acc = 0
                    for _ in range(test_data_size // test_batch_size + 1):
                        test_loss_batch, test_acc_batch = sess.run([test_model.loss,
                                                         test_model.accuracy])
                        test_loss += test_loss_batch
                        test_acc += test_acc_batch

                    test_loss = test_loss / test_data_size * test_batch_size
                    test_acc = test_acc / test_data_size

                    minutes = (now - prev) / 60
                    result = 'global_step: {} | trn_loss : {} trn_acc : {} test loss : {} test acc : {}'.format(global_step, trn_loss,
                                                    trn_acc, test_loss, test_acc)
                    message = 'took {} min, running at {} samples / min'.format(minutes,
                        batch_size * 5000 / minutes)
                    print(model_dir)
                    print(result)
                    print(message)
                    terminal_output.write(result+'\n')
                    terminal_output.write(message+'\n')
                    terminal_output.flush()

                    prev = time.time()

                    # print([idx_to_value['answer'][x] for x in trn_pred])
                    # print([idx_to_value['answer'][x] for x in test_pred])
                    saver.save(sess, model_dir+'/model.ckpt', global_step=global_step)

        except tf.errors.OutOfRangeError:
            print('Done training --epoch limit reached')
        finally:
            coord.request_stop()
        coord.join(threads)
        terminal_output.close()
