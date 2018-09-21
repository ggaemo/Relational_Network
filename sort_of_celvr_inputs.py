import tensorflow as tf
import h5py
import os
import numpy as np
import vqa_util
import pickle

def make_tf_record_file(data_type):

    def make_example(img, qst_color, qst_type, qst_subtype, answer):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        features = tf.train.Features(feature={'img_raw':_bytes_feature([img]),
                                              'answer': _int64_feature([answer]),
                                              'question_color': _int64_feature(
                                                  [qst_color]),
                                              'question_type': _int64_feature(
                                                  [qst_type]),
                                              'question_subtype': _int64_feature(
                                                  [qst_subtype]),
                                              })

        example = tf.train.Example(features = features)

        return example


    if not os.path.exists('data/Sort-of-CLEVR/seq_tfrecord_data/{0}'.format(data_type)):
        os.makedirs('data/Sort-of-CLEVR/seq_tfrecord_data/{0}'.format(data_type))

    if len(os.listdir('data/Sort-of-CLEVR/seq_tfrecord_data/{0}'.format(data_type))) == 0:

        writer = tf.python_io.TFRecordWriter('data/Sort-of-CLEVR/seq_tfrecord_data/{'
                                             '0}/{0}.tfrecord'.format(data_type))

        dirs = 'data/Sort-of-CLEVR/raw_data'
        filename = os.path.join(dirs, 'sort-of-clevr.pickle')
        with  open(filename, 'rb') as f:
            trn, test = pickle.load(f)
            if data_type == 'train':
                data = trn
            elif data_type == 'val':
                data = test


        print('datasets saved at {}'.format(filename))
        for val in data:
            image, relation_qa, nonrelation_qa = val
            image = image.astype(np.uint8)
            image = image.tostring()
            relation_q, relation_a = relation_qa
            nonrelation_q, nonrelation_a = nonrelation_qa
            ##6 for one-hot vector of color, 2 for question type, 3 for question subtype
            relation_q_list = [np.where(x)[0] for x in relation_q]
            nonrelation_q_list = [np.where(x)[0] for x in nonrelation_q]

            q_list = relation_q_list + nonrelation_q_list
            q_list = [(x[0], x[1] - 6, x[2] - 8) for x in q_list]

            q_color_list = [x[0] for x in q_list]
            q_type_list = [x[1] for x in q_list]
            q_subtype_list = [x[2] for x in q_list]
            a_list = relation_a + nonrelation_a

            for q_color, q_type, q_subtype, a in zip(q_color_list, q_type_list ,
                                                     q_subtype_list, a_list):
                ex = make_example(image, q_color, q_type, q_subtype, a)
                writer.write(ex.SerializeToString())
        writer.close()




        # data = h5py.File(data_path, 'r')
        # for key, val in data.items():
        #     question = np.where(val['question'].value)[0]
        #     question[1] = question[1] - vqa_util.NUM_COLOR
        #     image = val['image'].value.tostring()
        #     answer = np.where(val['answer'].value)[0]
        #     ex = make_example(image, question, answer)
        #     writer.write(ex.SerializeToString())
        # writer.close()
        print('tfrecord {} made'.format(data_type))

    else:
        print('tfrecord already made')

def inputs(batch_size, num_parallel_calls=10):

    def decode(serialized_example):
        """Parses an image and label from the given `serialized_example`."""

        parsed = tf.parse_single_example(
            serialized_example,
            features={'img_raw': tf.FixedLenFeature([], tf.string),
                      'answer': tf.FixedLenFeature([], tf.int64),
                      'question_color' : tf.FixedLenFeature([], tf.int64),
                      'question_type': tf.FixedLenFeature([], tf.int64),
                      'question_subtype': tf.FixedLenFeature([], tf.int64),
        })

        image = tf.decode_raw(parsed['img_raw'], tf.uint8)
        image = tf.reshape(image, [75, 75, 3])

        # image = tf.image.resize_images(image, (128, 128), method=1) # nearest neighbor

        image = tf.cast(image, tf.float32)
        image = (image - 127.5) / 127.5

        question_color = tf.cast(parsed['question_color'], tf.int32)
        question_type = tf.cast(parsed['question_type'], tf.int32)
        question_subtype = tf.cast(parsed['question_subtype'], tf.int32)

        answer = tf.cast(parsed['answer'], tf.int32)

        return {'img': image, 'qst_c': question_color, 'qst_type': question_type,
                'qst_subtype': question_subtype, 'ans': answer}

    def make_dataset(file_list):
        dataset = tf.data.TFRecordDataset(file_list)
        dataset = dataset.map(decode, num_parallel_calls=num_parallel_calls)
        # dataset = dataset.filter(lambda x: tf.reshape(tf.less(x['qst_len'], 10), []))
        dataset = dataset.prefetch(batch_size * 2)
        dataset = dataset.shuffle(buffer_size = batch_size * 10)
        dataset = dataset.batch(batch_size)

        return dataset

    for data_type in ['train', 'val']:
        dir_path = 'data/Sort-of-CLEVR/seq_tfrecord_data/{}/'.format(data_type)

        if len(os.listdir(dir_path)) == 0:
            make_tf_record_file(data_type)

        files_path = [dir_path+x for x in os.listdir(dir_path)]

        if data_type =='train':
            trn_dataset = make_dataset(files_path)
        elif data_type =='val':
            test_dataset = make_dataset(files_path)

    iterator = tf.data.Iterator.from_structure(trn_dataset.output_types, trn_dataset.output_shapes)

    next_batch = iterator.get_next()

    trn_init_op = iterator.make_initializer(trn_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    return next_batch, trn_init_op, test_init_op




def test():
    import matplotlib.pyplot as plt

    with tf.Session() as sess:
        batch_size = 32
        next_batch, trn_init_op, test_init_op = inputs(batch_size)

        # with open('data/CLEVR_v1.0/processed_data/question_answer_dict.pkl', 'rb') as f:
        #     convert_dict = pickle.load(f)
        # word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word = convert_dict

        while True:
            sess.run(trn_init_op)

            a = sess.run(next_batch)

            idx_to_word[95] = 'START'
            idx_to_word[96] = 'END'
            idx_to_word[0] = '_'
            print('train')

            for i in range(batch_size):
                print([idx_to_word[x] for x in a['qst'][i]])
                print([answer_idx_to_word[x] for x in a['ans'][i]])

                # plt.imshow(np.asarray(a['img'][i], np.uint8))
                # plt.imshow(a['img'][i])
                plt.show()
                response = input('next')
                if response == 'n':
                    continue
                else:
                    break

            print('test')
            sess.run(test_init_op)

            a = sess.run(next_batch)

            for i in range(batch_size):
                # print([idx_to_word[x] for x in a['qst'][i]])
                # print([answer_idx_to_word[x] for x in a['ans'][i]])
                plt.imshow(np.asarray(a['img'][i]), np.uint8)
                # plt.imshow(a['img'][i])
                plt.show()
                response = input('next')
                if response == 'n':
                    continue
                else:
                    break

if __name__ == '__main__':
    test()