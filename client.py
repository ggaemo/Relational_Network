import model
import inputs
import tensorflow as tf
import pickle
import time
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-batch_size', type=int)
parser.add_argument('-test_data_size', type=int, default=149991)
parser.add_argument('-test_batch_size', type=int, default=128, help='second biggest '
                                                                    'denominator')
parser.add_argument('-num_epochs', type=int, default=10000)
parser.add_argument('-word_embedding_size', type=int)
parser.add_argument('-rnn_hidden_dim', type=int)
parser.add_argument('-g_theta_layers', type=int, nargs='+')
parser.add_argument('-f_phi_layers', type=int, nargs='+')
parser.add_argument('-img_encoding_layers', type=int, nargs='+')
parser.add_argument('-option', type=str, default='org')
parser.add_argument('-restore', action='store_true', default=False)
args = parser.parse_args()


batch_size = args.batch_size
test_data_size = args.test_data_size
test_batch_size = args.test_batch_size
num_epochs = args.num_epochs
word_embedding_size = args.word_embedding_size
rnn_hidden_dim = args.rnn_hidden_dim
img_encoding_layers = args.img_encoding_layers
g_theta_layers = args.g_theta_layers # [512, 512, 512]
f_phi_layers = args.f_phi_layers # [512, 1024]
option = args.option
restore = args.restore


img_encoding_layers_parsed = [img_encoding_layers[i:i+3] for i in
                              np.arange(len(img_encoding_layers), step =3)]

model_dir = 'model/{}_{}_{}_{}_{}_{}/'.format(word_embedding_size,
                                              rnn_hidden_dim,
                                              '-'.join([str(x) for x in g_theta_layers]),
                                              '-'.join([str(x) for x in f_phi_layers]),
                                              '-'.join([str(x) for x in img_encoding_layers]),
                                              option)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
elif not restore:
    print('directory exists')
    raise FileExistsError

with open(os.path.join(model_dir, 'model_config.txt'), 'w') as f:
    f.write('batch size : {}'.format(batch_size))
    f.write('num epoch : {}'.format(num_epochs))
    f.write('word_embedding size: {}'.format(word_embedding_size))
    f.write('rnn_hidden_dim: {}'.format(rnn_hidden_dim))
    f.write('g_theta layers: {}'.format(g_theta_layers))
    f.write('f_phi_layers: {}'.format(f_phi_layers))



terminal_output = open(model_dir+'terminal_output.txt', 'w')


with open('data/CLEVR_v1.0/processed_data/idx_to_value.pkl', 'rb') as f:
    idx_to_value = pickle.load(f)

with open('data/CLEVR_v1.0/processed_data/question_answer_dict.pkl', 'rb') as f:
    word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word = pickle.load(f)

idx_to_value['question'] = idx_to_word
idx_to_value['answer'] = answer_idx_to_word

with tf.Graph().as_default():
    next_batch, trn_init_op, test_init_op = inputs.inputs(batch_size)

    with tf.variable_scope('Model', reuse=None):

        model = model.RelationalNetwork(next_batch, idx_to_value,
                                        word_embedding_size, g_theta_layers,
                                        f_phi_layers,
                                        img_encoding_layers_parsed,
                                        rnn_hidden_dim)



    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(model_dir, flush_secs=5)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth =True
    with tf.Session(config=config) as sess:
        global_step = 0
        if restore:
            latest_model = tf.train.latest_checkpoint(model_dir)
            print('restored model from ', latest_model)
            saver.restore(sess, latest_model)
        else:
            sess.run(tf.global_variables_initializer())

        for i in range(num_epochs):
            print('epoch num', i)
            prev = time.time()
            sess.run(trn_init_op)
            sess.run(tf.local_variables_initializer())
            try:
                while True:

                    if global_step % 3000 == 0:
                        _, global_step, trn_loss_summary = sess.run([model.train_op,
                                                          model.global_step,
                                                   model.trn_loss_summary])
                        summary_writer.add_summary(trn_loss_summary, i)

                        # pred = sess.run(model.prediction)
                        # print('train pred', pred)
                    else:
                        _, global_step = sess.run([model.train_op, model.global_step])

            except tf.errors.OutOfRangeError:
                print('out of range')
                now = time.time()
                summary_value, trn_acc = sess.run([model.summary_trn,
                                                   model.accuracy])
                summary_writer.add_summary(summary_value, global_step=i)

                sess.run(test_init_op)
                sess.run(tf.local_variables_initializer()) # metrics value init to 0

                try:
                    print('test_start')
                    tmp_step = 0
                    while True:
                        if tmp_step % 3000 == 0:
                            _, test_loss_summary = sess.run([model.update_ops,
                                                 model.test_loss_summary])
                            print('model accruacy test', sess.run(model.accuracy))
                            summary_writer.add_summary(test_loss_summary, global_step=i)

                            # pred = sess.run(model.prediction)
                            # print('test pred', pred)

                        else:
                            sess.run(model.update_ops)
                        tmp_step += 1

                except tf.errors.OutOfRangeError:
                    print('test_start end')
                    summary_value, test_acc = sess.run([model.summary_test,
                                                       model.accuracy])
                    summary_writer.add_summary(summary_value, global_step=i)



                minutes = (now - prev) / 60
                result = 'num iter: {} | trn_acc : {} test acc : {}'.format(
                    global_step, trn_acc, test_acc)

                message = 'took {} min'.format(minutes)
                print(model_dir)
                print(result)
                print(message)
                terminal_output.write(result + '\n')
                terminal_output.write(message + '\n')
                terminal_output.flush()

                # print([idx_to_value['answer'][x] for x in trn_pred])
                # print([idx_to_value['answer'][x] for x in test_pred])
                saver.save(sess, model_dir + '/model.ckpt', global_step=i)

        terminal_output.close()