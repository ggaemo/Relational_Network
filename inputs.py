import time
import pandas as pd
import numpy as np

import pickle
import os, json, re, itertools, collections
import tensorflow as tf
import os

def make_question_data_pickle(data_type):
    if not os.path.exists('data/CLEVR_v1.0/processed_data'):
        os.makedirs('data/CLEVR_v1.0/processed_data')

    if os.path.exists('data/CLEVR_v1.0/processed_data/qa_data_{}.pkl'.format(data_type)):
        print('loaded qa_data_{}.pkl'.format(data_type))

        with open('data/CLEVR_v1.0/processed_data/qa_data_{}.pkl'.format(data_type), 'rb') as f:
            qa_data = pickle.load(f)
    else:
        print('processing question_data_{}.pkl'.format(data_type))
        data_path = 'data/CLEVR_v1.0/questions/CLEVR_{}_questions.json'.format(data_type)

        if data_type == 'train':
            qa_data, word_to_idx, idx_to_word, answer_word_to_idx, \
            answer_idx_to_word = read_and_preprocess_question_data(data_path)
            with open('data/CLEVR_v1.0/processed_data/question_answer_dict.pkl', 'wb') as f:
                pickle.dump(
                    [word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word], f)
        else:
            with open('data/CLEVR_v1.0/processed_data/question_answer_dict.pkl', 'rb') as f:
                convert_dict = pickle.load(f)
            qa_data, word_to_idx, idx_to_word, answer_word_to_idx, \
            answer_idx_to_word = read_and_preprocess_question_data(data_path, convert_dict)

        with open('data/CLEVR_v1.0/processed_data/qa_data_{}.pkl'.format(data_type), 'wb') as f:
            pickle.dump(qa_data, f)

    return qa_data

def read_and_preprocess_question_data(data_path, convert_dict = None):

    d = json.loads(open(data_path).read())
    word_set = set()
    answer_word_set = set()
    qa_data = collections.defaultdict(list)

    for q_obj in d['questions']:
        img_idx = q_obj['image_index']
        q_text = q_obj['question'].lower()
        q_text = re.sub('\s+', ' ', q_text)
        q_text_without_question_mark = q_text[:-1]
        q_words = q_text_without_question_mark.split(' ')
        word_set.update(q_words)

        a_text = q_obj['answer'].lower()
        a_text = re.sub('\s+', ' ', a_text)
        answer_word_set.add(a_text)

        qa_data[img_idx].append((q_words, a_text))

    if convert_dict:
        word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word = convert_dict
    else:
        word_to_idx = dict()
        idx_to_word = dict()
        for idx, word in enumerate(word_set, start=1):
            # index starts with 1 because 0 is used as the padded value when batches are
            #  created
            word_to_idx[word] = idx
            idx_to_word[idx] = word

        answer_word_to_idx = dict()
        answer_idx_to_word = dict()
        for idx, word in enumerate(answer_word_set, start=0):
            # single answer, so no padded values of 0 are created. thus index starts with 0
            answer_word_to_idx[word] = idx
            answer_idx_to_word[idx] = word

    qa_idx_data = collections.defaultdict(list)
    for img_idx, qa_list in qa_data.items():
        for q_word_list, answer_word in qa_list:
            q = [word_to_idx[word] for word in q_word_list]
            a = answer_word_to_idx[answer_word]
            qa_idx_data[img_idx].append((q, a))


    return qa_idx_data, word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word

def make_seq_tf_record_file(data_type):

    def make_example(img, qst, answer):

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

        question_features = [_int64_feature([x]) for x in qst]

        feature_list = {'question': tf.train.FeatureList(feature=question_features)}

        feature_lists = tf.train.FeatureLists(feature_list=feature_list)

        context_features = tf.train.Features(feature={'img_raw':_bytes_feature([img]),
                                                      'answer': _int64_feature([answer]),
                                                      'question_len':_int64_feature([len(qst)])
        })

        example = tf.train.SequenceExample(feature_lists=feature_lists,
                                           context=context_features)

        return example

    qa_data = make_question_data_pickle(data_type)

    img_files = os.listdir('data/CLEVR_v1.0/images/{}'.format(data_type))

    num_shards = 20
    num_img_per_shard = int(len(img_files) / num_shards)
    if not os.path.exists('data/CLEVR_v1.0/seq_tfrecord_data/{0}'.format(data_type)):
        os.makedirs('data/CLEVR_v1.0/seq_tfrecord_data/{0}'.format(data_type))

    if len(os.listdir('data/CLEVR_v1.0/seq_tfrecord_data/{0}'.format(data_type))) == 0:
        for shard_num in range(num_shards):
            writer = tf.python_io.TFRecordWriter('data/CLEVR_v1.0/seq_tfrecord_data/{'
                                                 '0}/{0}_{1}.tfrecord'.format(data_type,
                                                                          shard_num))
            print('shard num {} writing'.format(shard_num) )
            start_idx = num_img_per_shard * shard_num
            end_idx = min(num_img_per_shard * (shard_num + 1), len(img_files))
            print(start_idx, end_idx)
            for idx in range(start_idx, end_idx):
                img_file = img_files[idx]
                img_file_path = 'data/CLEVR_v1.0/images/{}/{}'.format(data_type, img_file)
                img_data = tf.gfile.FastGFile(img_file_path, 'rb').read()
                img_idx = int(re.search('CLEVR_{}_(\d+).png'.format(data_type), img_file).group(1))
                for question, answer in qa_data[img_idx]:
                    ex = make_example(img_data, question, answer)
                    writer.write(ex.SerializeToString())
            writer.close()
        print('tfrecord {} made'.format(data_type))

    else:
        print('tfrecord already made')

def inputs(batch_size, num_parallel_calls=10):

    def decode(serialized_example):
        """Parses an image and label from the given `serialized_example`."""

        context_parsed, sequence_parsed = tf.parse_single_sequence_example(
            serialized_example,
            context_features={'img_raw': tf.FixedLenFeature([], tf.string),
                'answer': tf.FixedLenFeature([], tf.int64),
                              'question_len':tf.FixedLenFeature([], tf.int64)},
            sequence_features={
                'question': tf.FixedLenSequenceFeature([], tf.int64)})

        image = tf.image.decode_png(context_parsed['img_raw'],channels=3)

        image = tf.image.resize_images(image, (128, 128), method=1) # nearest neighbor

        image = tf.cast(image, tf.float32)

        # question = tf.cast(sequence_parsed['question'], tf.int32)
        question = sequence_parsed['question']
        answer = tf.expand_dims(tf.cast(context_parsed['answer'], tf.int32), axis=0)
        question_len: answer = tf.expand_dims(tf.cast(context_parsed['question_len'], tf.int32), axis=0)

        return {'img': image, 'qst': question, 'ans': answer,
        'qst_len':question_len}

    def make_dataset(file_list):
        dataset = tf.data.TFRecordDataset(file_list)
        dataset = dataset.map(decode, num_parallel_calls=num_parallel_calls)
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.shuffle(buffer_size = batch_size * 4)
        dataset = dataset.padded_batch(batch_size,
                                       padded_shapes={'img': tf.TensorShape([None, None,3                                     ]),
                                                      'ans' : 1,
                                       'qst':tf.TensorShape([None]),
                                                      'qst_len':1})

        return dataset

    for data_type in ['train', 'val']:
        dir_path = 'data/CLEVR_v1.0/seq_tfrecord_data/{}/'.format(data_type)

        if len(os.listdir(dir_path)) == 0:
            make_seq_tf_record_file(data_type)

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
    with tf.Session() as sess:
        next_batch, trn_init_op = inputs('val', 20)

        sess.run(trn_init_op)



        while True:
            a = sess.run(next_batch)
            import matplotlib.pyplot as plt
            plt.imshow(a['img'])
            print(len(a))


# make_seq_tf_record_file('train')
# make_seq_tf_record_file('val')

if __name__ =='__main__':
    test()


