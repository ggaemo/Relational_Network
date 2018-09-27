import cv2
import re
import tensorflow as tf
import pickle
import time
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-data', type=str)
parser.add_argument('-model_type', type=str)
parser.add_argument('-batch_size', type=int)
parser.add_argument('-learning_rate', type=float, default=2.5*1e-4)
# parser.add_argument('-test_data_size', type=int, default=149991)
# parser.add_argument('-test_batch_size', type=int, default=128, help='second biggest '
#                                                                     'denominator')
parser.add_argument('-num_epochs', type=int, default=10000)
parser.add_argument('-word_embedding_size', type=int)
parser.add_argument('-rnn_hidden_dim', type=int)
parser.add_argument('-g_theta_layers', type=int, nargs='+')
parser.add_argument('-f_phi_layers', type=int, nargs='+')
parser.add_argument('-img_encoding_layers', type=int, nargs='+')
parser.add_argument('-option', type=str, default='org')
parser.add_argument('-run_meta', action='store_true', default=False)
parser.add_argument('-restore', action='store_true', default=False)
args = parser.parse_args()


data = args.data
model_type = args.model_type
batch_size = args.batch_size
num_epochs = args.num_epochs
word_embedding_size = args.word_embedding_size
rnn_hidden_dim = args.rnn_hidden_dim
img_encoding_layers = args.img_encoding_layers
g_theta_layers = args.g_theta_layers # [512, 512, 512]
f_phi_layers = args.f_phi_layers # [512, 1024]
option = args.option
run_meta = args.run_meta
restore = args.restore
base_learning_rate = args.learning_rate


img_encoding_layers_parsed = [img_encoding_layers[i:i+3] for i in
                              np.arange(len(img_encoding_layers), step =3)]

model_dir = 'model/{}/{}_bs-{}_we-{}_rnn-{}_g-{}_f-{}_cnn-{}_{}/'.format(data,
                                                                         model_type, batch_size,
                                                                   word_embedding_size,
                                                                   rnn_hidden_dim,
                                              '-'.join([str(x) for x in g_theta_layers]),
                                              '-'.join([str(x) for x in f_phi_layers]),
                                              '-'.join([str(x) for x in img_encoding_layers]),
                                              option)

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
elif 'checkpoint' in os.listdir(model_dir) and not restore:
    print('saved model exists')
    raise FileExistsError



with open(os.path.join(model_dir, 'model_config.txt'), 'w') as f:
    f.write('batch size : {}'.format(batch_size))
    f.write('num epoch : {}'.format(num_epochs))
    f.write('word_embedding size: {}'.format(word_embedding_size))
    f.write('rnn_hidden_dim: {}'.format(rnn_hidden_dim))
    f.write('g_theta layers: {}'.format(g_theta_layers))
    f.write('f_phi_layers: {}'.format(f_phi_layers))



terminal_output = open(model_dir+'terminal_output.txt', 'w')



with tf.Graph().as_default():


    if data == 'CLEVR':
        import model
        import inputs

        with open('data/CLEVR_v1.0/processed_data/question_answer_dict.pkl', 'rb') as f:
            word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word = pickle.load(f)

            idx_to_word[0] = '_'
            qst_vocab_size = len(idx_to_word)
            idx_to_word[qst_vocab_size] = 'START'
            idx_to_word[qst_vocab_size+1] = 'END'

            word_to_idx['_'] = 0
            word_to_idx['START'] = qst_vocab_size
            word_to_idx['END'] = qst_vocab_size + 1


            qst_vocab_size = len(word_to_idx)
            ans_vocab_size = len(answer_word_to_idx)


            print('START AND END TOKEN ADDED TO QUESTION VOCAB')

        with tf.variable_scope('inputs'):
            next_batch, trn_init_op, test_init_op = inputs.inputs(batch_size)

        with tf.variable_scope('Model', reuse=None):

            model = model.RelationalNetwork(next_batch, qst_vocab_size, ans_vocab_size,
                                            word_embedding_size, g_theta_layers,
                                            f_phi_layers,
                                            img_encoding_layers_parsed,
                                            rnn_hidden_dim, batch_size=batch_size,
                                            base_learning_rate=base_learning_rate)

        save_interval = 3000

    elif data == 's_CLEVR':

        import sort_of_celvr_inputs as inputs
        import sort_of_clevr_generator_2

        # import vqa_util
        idx_to_qst_type = sort_of_clevr_generator_2.question_type_dict
        idx_to_ans = sort_of_clevr_generator_2.answer_dict
        idx_to_color = sort_of_clevr_generator_2.color_dict

        qst_color_vocab = len(idx_to_color)
        qst_type_vocab_size = len(idx_to_qst_type)
        ans_vocab_size = len(idx_to_ans)


        with tf.variable_scope('inputs'):
            next_batch, trn_init_op, test_init_op = inputs.inputs(batch_size)

        if model_type == 'rn':
            import model_sort_of_clevr

            model = model_sort_of_clevr.RelationalNetwork(next_batch, qst_color_vocab,
                                                          qst_type_vocab_size,
                                                          ans_vocab_size,
                                                          word_embedding_size, g_theta_layers,
                                                          f_phi_layers,
                                                          img_encoding_layers_parsed,
                                                          batch_size=batch_size,
                                                          question_type_dict=idx_to_qst_type,
                                                          base_learning_rate=base_learning_rate)
        elif model_type == 'base':
            import model_base
            model = model_base.RelationalNetwork(next_batch, qst_color_vocab,
                                                          qst_type_vocab_size,
                                                          ans_vocab_size,
                                                          word_embedding_size,
                                                          g_theta_layers,
                                                          f_phi_layers,
                                                          img_encoding_layers_parsed,
                                                          batch_size=batch_size,
                                                 question_type_dict=idx_to_qst_type,
                                                 base_learning_rate=base_learning_rate)

        save_interval = 1000

    config = tf.ConfigProto()
    config.gpu_options.allow_growth =True
    with tf.Session(config=config) as sess:
        saver = tf.train.Saver(max_to_keep=5)
        summary_writer = tf.summary.FileWriter(model_dir, flush_secs=5, graph=sess.graph)
        global_step = 0
        if restore:
            latest_model = tf.train.latest_checkpoint(model_dir)
            print('restored model from ', latest_model)
            epoch_num = int(re.search('model.ckpt-(\d+)', latest_model).group(1))
            sess.run(tf.assign(model.epoch, epoch_num))
            saver.restore(sess, latest_model)
        else:
            sess.run(tf.global_variables_initializer())
            epoch_num = sess.run(model.epoch)

        for _ in range(num_epochs):
            print('epoch num', epoch_num, 'batch iteration', global_step)
            prev = time.time()
            sess.run(trn_init_op)
            sess.run(tf.local_variables_initializer())
            try:

                while True:

                    if restore:
                        while True:
                            sess.run(model.ans, feed_dict={model.is_training:True})

                    # question_embed, qst_len, rnn_outputs, last_states = sess.run(model.get)

                    # if global_step % save_interval * 10== 0:
                    #     img, pred, ans, qst = sess.run([model.img, model.prediction,
                    #                                 model.ans, model.qst],
                    #                                    feed_dict={model.is_training:True})
                    #     img = img[:10]
                    #     pred = pred[:10]
                    #     ans = ans[:10]
                    #     qst = qst[:10]
                    #
                    #     if data == 'CLEVR':
                    #
                    #         ans = [answer_idx_to_word[x] for x in np.squeeze(ans)]
                    #         qst = [' '.join([idx_to_word[x] for x in row]) for row in qst]
                    #         pred = [answer_idx_to_word[x] for x in np.squeeze(pred)]
                    #
                    #     elif data =='s_CLEVR':
                    #         ans = [idx_to_ans[x] for x in ans]
                    #         qst = ['{}_{}'.format(idx_to_color[x], idx_to_qst_type[y])
                    #                for x, y in qst]
                    #         pred = [idx_to_ans[x] for x in pred]
                    #
                    #     summary = sess.run(model.summary_additional, {model.img_pl:img,
                    #                                         model.ans_word: ans,
                    #                                         model.qst_word: qst,
                    #                                         model.pred_word:pred})
                    #     summary_writer.add_summary(summary, global_step=epoch_num)

                        # flat_1, flat_1_low, flat_2, flat_2_low, encoded_img_pair, \
                        # encoded_img_qst_pair, encoded_img_qst_pair_tp, pair_output_sum \
                        #     = sess.run(model.get, {model.is_training:True})
                    if run_meta:
                        if global_step % save_interval == 0:
                            print('run_meta')
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()

                            _, global_step, trn_loss_summary = sess.run([model.train_op,
                                                                         model.global_step,
                                                                         model.trn_loss_summary],
                                                                        {model.is_training: True},
                                                                        options=run_options,
                                                                        run_metadata=run_metadata)
                            summary_writer.add_summary(trn_loss_summary, i)
                            summary_writer.add_run_metadata(run_metadata, 'step_{}'.format(
                                epoch_num),
                                                            epoch_num)



                    if global_step % save_interval == 0:
                        _, global_step, trn_loss_summary = sess.run([model.train_op,
                                                          model.global_step,
                                                   model.trn_loss_summary],
                                                                    {model.is_training:True})

                        summary_writer.add_summary(trn_loss_summary, global_step)
                    else:
                        _, global_step = sess.run([model.train_op, model.global_step],
                                                                    {model.is_training:True})



            except tf.errors.OutOfRangeError:
                sess.run(model.increment_epoch_op)
                epoch_num = sess.run(model.epoch)
                print('out of range', 'epoch', epoch_num, 'iter', global_step)
                now = time.time()
                summary_value, trn_acc = sess.run([model.summary_trn,
                                                   model.accuracy],
                                                                    {model.is_training:False})
                summary_writer.add_summary(summary_value, global_step=epoch_num)

                sess.run(test_init_op)
                sess.run(tf.local_variables_initializer()) # metrics value init to 0

                try:
                    print('test_start')
                    tmp_step = 0

                    img, pred, ans, qst = sess.run([model.img, model.prediction,
                                                model.ans, model.qst],
                                                   feed_dict={model.is_training:True})
                    sample_num = 10
                    img = img[:sample_num]
                    pred = pred[:sample_num]
                    ans = ans[:sample_num]
                    qst = qst[:sample_num]

                    if data == 'CLEVR':

                        ans = [answer_idx_to_word[x] for x in np.squeeze(ans)]
                        qst = [' '.join([idx_to_word[x] for x in row]) for row in qst]
                        pred = [answer_idx_to_word[x] for x in np.squeeze(pred)]

                    elif data =='s_CLEVR':
                        img = np.pad(img, [(0, 0), (0, 20), (0, 0), (0, 0)],
                                            mode='constant', constant_values=(1, 1))
                        ans = [idx_to_ans[x] for x in ans]
                        qst = ['{}_{}'.format(idx_to_color[x], idx_to_qst_type[y])
                               for x, y in qst]
                        pred = [idx_to_ans[x] for x in pred]

                        print(ans)
                        print(pred)

                        for i in range(sample_num):
                            cv2.line(img[i], (37, 0), (37, 75), (0, 0, 0), 1)
                            cv2.line(img[i], (0, 37), (75, 37), (0, 0, 0), 1)
                            cv2.line(img[i], (0, 75), (75, 75), (0, 0, 0), 1)
                            cv2.putText(img[i], '{} {} {}'.format(qst[i], ans[i], pred[i]),
                                        (2, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

                    summary = sess.run(model.summary_additional, {model.img_pl:img,
                                                        model.ans_word: ans,
                                                        model.qst_word: qst,
                                                        model.pred_word:pred})
                    summary_writer.add_summary(summary, global_step=epoch_num)


                    while True:
                        if tmp_step % save_interval == 0:
                            _, test_loss_summary = sess.run([model.summary_update_ops,
                                                 model.test_loss_summary],
                                                                    {
                                                                        model.is_training:False})
                            print('accuracy batch test', sess.run(model.accuracy))
                            summary_writer.add_summary(test_loss_summary, global_step=epoch_num)
                        else:
                            sess.run(model.summary_update_ops, {model.is_training:False})

                        tmp_step += 1

                except tf.errors.OutOfRangeError:
                    print('test_start end')
                    summary_value, test_acc = sess.run([model.summary_test,
                                                       model.accuracy],
                                                                    {
                                                                        model.is_training:False})
                    summary_writer.add_summary(summary_value, global_step=epoch_num)



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
                saver.save(sess, os.path.join(model_dir, 'model.ckpt'),
                           global_step=epoch_num)


        terminal_output.close()