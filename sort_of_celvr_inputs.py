import tensorflow as tf
import os
import numpy as np
import pickle

def make_tf_record_file(data_type, data_option):

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

    data_dir = 'data/Sort-of-CLEVR/seq_tfrecord_data'
    if data_option:
        data_dir = os.path.join(data_dir, data_option)

    if not os.path.exists(data_dir):
        os.makedirs(os.path.join(data_dir))

    filename = os.path.join(data_dir, '{}.tfrecord'.format(data_type))
    if not os.path.exists(filename):

        writer = tf.python_io.TFRecordWriter(filename)

        dirs = 'data/Sort-of-CLEVR/raw_data'
        if data_option:
            dirs = os.path.join(dirs, data_option)

        raw_data_filename = os.path.join(dirs, 'sort-of-clevr.pickle')

        if not os.path.exists(raw_data_filename):
            import sort_of_clevr_generator_2
            sort_of_clevr_generator_2.generate_data(data_option)

        with  open(raw_data_filename, 'rb') as f:
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

def inputs(batch_size, data_option, num_parallel_calls=10):

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

        # image = (image + 1.0) / 2.0

        question_color = tf.cast(parsed['question_color'], tf.int32)
        question_type = tf.cast(parsed['question_type'], tf.int32)
        question_subtype = tf.cast(parsed['question_subtype'], tf.int32)

        answer = tf.cast(parsed['answer'], tf.int32)

        return {'img': image, 'qst_c': question_color, 'qst_type': question_type,
                'qst_subtype': question_subtype, 'ans': answer}

    def make_dataset(file_list, data_type):
        dataset = tf.data.TFRecordDataset(file_list)


        dataset = dataset.map(decode, num_parallel_calls=num_parallel_calls)
        # dataset = dataset.filter(lambda x: tf.reshape(tf.less(x['qst_len'], 10), []))

        if data_type == 'train':
            dataset = dataset.shuffle(buffer_size = batch_size * 10)
        dataset = dataset.batch(batch_size)

        dataset = dataset.prefetch(batch_size * 10)

        return dataset

    for data_type in ['train', 'val']:
        data_dir = 'data/Sort-of-CLEVR/seq_tfrecord_data/{}'.format(data_option)

        file_path = '{}/{}.tfrecord'.format(data_dir, data_type)

        if not os.path.exists(file_path):

            make_tf_record_file(data_type, data_option)

        # files_path = [dir_path+x for x in os.listdir(dir_path)]

        if data_type =='train':
            trn_dataset = make_dataset(file_path, data_type)
        elif data_type =='val':
            test_dataset = make_dataset(file_path, data_type)

    iterator = tf.data.Iterator.from_structure(trn_dataset.output_types, trn_dataset.output_shapes)

    next_batch = iterator.get_next()

    trn_init_op = iterator.make_initializer(trn_dataset)
    test_init_op = iterator.make_initializer(test_dataset)

    with open('data/Sort-of-CLEVR/raw_data/{}/ans_color_qst_dict.pickle'.format(data_option),
              'rb') as f:
        answer_dict, color_dict, question_type_dict = pickle.load(f)

    return next_batch, trn_init_op, test_init_op, answer_dict, color_dict, question_type_dict




def test():
    import matplotlib.pyplot as plt
    import cv2
    with tf.Session() as sess:
        batch_size = 24
        next_batch, trn_init_op, test_init_op = inputs(batch_size)

        # with open('data/CLEVR_v1.0/processed_data/question_answer_dict.pkl', 'rb') as f:
        #     convert_dict = pickle.load(f)
        # word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word = convert_dict

        import sort_of_clevr_generator_2

        color_dict = sort_of_clevr_generator_2.color_dict
        question_type_dict = sort_of_clevr_generator_2.question_type_dict
        answer_dict = sort_of_clevr_generator_2.answer_dict

        while True:
            sess.run(test_init_op)

            a = sess.run(next_batch)

            qst_c = a['qst_c']

            qst = a['qst_type'] * 3 + a['qst_subtype']

            ans = a['ans']

            img = a['img']

            height = img.shape[1]
            half = int(height / 2)

            print(half)


            for i in range(batch_size):

                if qst[i] == 1 or qst[i] == 2:

                    print(color_dict[qst_c[i]], question_type_dict[qst[i]])
                    print(answer_dict[ans[i]])
                    cv2.line(img[i], (half, 0), (half, height), (0, 0, 0), 1)
                    cv2.line(img[i], (0, half), (height, half), (0, 0, 0), 1)
                    plt.imshow(img[i])
                    # plt.imshow(np.asarray(img[i], np.uint8))
                    # plt.imshow(a['img'][i])
                    plt.show()

                    response = input('next')
                    if response == 'n':
                        continue
                    else:
                        break


            # idx_to_word[95] = 'START'
            # idx_to_word[96] = 'END'
            # idx_to_word[0] = '_'
            # print('train')
            #
            # for i in range(batch_size):
            #     print([idx_to_word[x] for x in a['qst'][i]])
            #     print([answer_idx_to_word[x] for x in a['ans'][i]])
            #
            #     # plt.imshow(np.asarray(a['img'][i], np.uint8))
            #     # plt.imshow(a['img'][i])
            #     plt.show()
            #     response = input('next')
            #     if response == 'n':
            #         continue
            #     else:
            #         break
            #
            # print('test')
            # sess.run(test_init_op)
            #
            # a = sess.run(next_batch)
            #
            # for i in range(batch_size):
            #     # print([idx_to_word[x] for x in a['qst'][i]])
            #     # print([answer_idx_to_word[x] for x in a['ans'][i]])
            #     plt.imshow(np.asarray(a['img'][i]), np.uint8)
            #     # plt.imshow(a['img'][i])
            #     plt.show()
            #     response = input('next')
            #     if response == 'n':
            #         continue
            #     else:
            #         break

if __name__ == '__main__':
    test()
