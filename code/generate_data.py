import pickle
import tensorflow as tf
import os, json, re, itertools, collections
import numpy as np
def read_and_preprocess_question_data(data_path, convert_dict = None):
    d = json.loads(open(data_path).read())
    word_set = set()
    text_data = list()
    answer_word_set = set()
    answer_text_data = list()
    for q_obj in d['questions']:
        q_text = q_obj['question'].lower()
        q_text = re.sub('\s+', ' ', q_text)
        q_text_without_question_mark = q_text[:-1]
        words = q_text_without_question_mark.split(' ')
        words.append('?')
        text_data.append(words)
        word_set.update(words)

        a_text = q_obj['answer'].lower()
        a_text = re.sub('\s+', ' ', a_text)
        answer_text_data.append(a_text)
        answer_word_set.add(a_text)

    if convert_dict:
        word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word = convert_dict
    else:
        word_to_idx = dict()
        idx_to_word = dict()
        for idx, word in enumerate(word_set, start=1):
            word_to_idx[word] = idx
            idx_to_word[idx] = word

        answer_word_to_idx = dict()
        answer_idx_to_word = dict()
        for idx, word in enumerate(answer_word_set, start=0):
            answer_word_to_idx[word] = idx
            answer_idx_to_word[idx] = word

    text_data_idx_form = list()
    for sentence in text_data:
        text_data_idx_form.append([word_to_idx[word] for word in sentence])

    answer_text_data_idx_form = list()
    for word in answer_text_data:
        answer_text_data_idx_form.append(answer_word_to_idx[word])

    return text_data_idx_form, answer_text_data_idx_form, word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word



def read_and_preprocess_input_data(data_path):
    scene = json.loads(open(data_path).read())

    data = collections.defaultdict(list)

    value_to_idx = dict()
    value_to_idx['color'] =  {'blue' : 1,
               'brown': 2,
               'cyan' : 3,
               'gray': 4,
               'green':5 ,
               'purple':6 ,
               'red': 7,
               'yellow': 8}
    value_to_idx['material'] = {'metal' : 1, 'rubber': 2}
    value_to_idx['shape'] = {'cube' : 1, 'cylinder': 2, 'sphere': 3}
    value_to_idx['size'] = {'large' : 1, 'small': 2}

    idx_to_value = dict()
    for key in value_to_idx:
        idx_to_value[key] = {idx: val for val, idx in value_to_idx[key].items()}

    for element in scene['scenes']:
        obj_list = element['objects']
        row = collections.defaultdict(list)
        len_key = len(obj_list)
        for obj1, obj2 in itertools.combinations(obj_list, 2):
            for key in obj1:
                tmp = list()
                if isinstance(obj1[key], list):
                    tmp.extend(obj1[key] + obj2[key])
                else:
                    if key in value_to_idx:
                        input_value = [value_to_idx[key][obj1[key]], value_to_idx[key][obj2[key]]]
                    else:
                        input_value = [obj1[key], obj2[key]]
                    tmp.extend(input_value)
                if key == '3d_coords':
                    key = 'xyz_coords'
                row[key].append(tmp)
        data[len_key].append(row)

    return data, idx_to_value


def make_example(xyz_coords, rotation, color, size, shape, material, pixel_coords, question, answer):

    xyz_coords = [np.array(x).tostring() for x in xyz_coords]
    rotation = [np.array(x).tostring() for x in rotation]
    color = [np.array(x).tostring() for x in color]
    size = [np.array(x).tostring() for x in size]
    shape = [np.array(x).tostring() for x in shape]
    material = [np.array(x).tostring() for x in material]
    pixel_coords = [np.array(x).tostring() for x in pixel_coords]
    question_word_len = len(question)

    ex = tf.train.SequenceExample()

    f_3c = ex.feature_lists.feature_list['xyz_coords']  # 개수는 object pair의 개수 만큼
    f_r = ex.feature_lists.feature_list['rotation']
    f_c = ex.feature_lists.feature_list['color']
    f_sh = ex.feature_lists.feature_list['shape']
    f_si = ex.feature_lists.feature_list['size']
    f_m = ex.feature_lists.feature_list['material']
    f_pc = ex.feature_lists.feature_list['pixel_coords']

    f_q = ex.feature_lists.feature_list['question']
    c_question_word_len = ex.context.feature['question_word_len']

    c_a = ex.context.feature['answer']

    c_obj_pair_num = ex.context.feature['num_pair']


    #         c_key_en_len = ex.context.feature['HW_KEY_EN_NM_LIST_LEN']
    #         c_en_len = ex.context.feature['HW_EN_NM_LIST_LEN']
    #         c_work_nm_len = ex.context.feature['WORK_NM_LEN']

    def add_seq(f, value_list, dtype):
        if dtype == 'int64':
            for val in value_list:
                f.feature.add().int64_list.value.append(val)
        elif dtype == 'float':
            for val in value_list:
                f.feature.add().float_list.value.append(val)
        elif dtype == 'bytes':
            for val in value_list:
                f.feature.add().bytes_list.value.append(val)
        return f

    def add_context(f, value):
        f.int64_list.value.append(value)

    f_3c = add_seq(f_3c, xyz_coords, 'bytes')
    f_r = add_seq(f_r, rotation, 'bytes')

    f_c = add_seq(f_c, color, 'bytes')
    f_si = add_seq(f_si, size, 'bytes')
    f_sh = add_seq(f_sh, shape, 'bytes')
    f_m = add_seq(f_m, material, 'bytes')
    f_pc = add_seq(f_pc, pixel_coords, 'bytes')

    f_q = add_seq(f_q, question, 'int64')
    c_question_word_len = add_context(c_question_word_len, question_word_len)

    c_a = add_context(c_a, answer)

    c_obj_pair_num = add_context(c_obj_pair_num, len(color)) # 특별히 color라서 의미 있는게 아니라,
    # 어떤거든... input에 들어가는 거면 써도댐

    return ex


def read_and_decode(filename_queue):
    print('Reading and Decoding')
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
        'num_pair' : tf.FixedLenFeature([], dtype=tf.int64),
        "answer": tf.FixedLenFeature([], dtype=tf.int64),
        "question_word_len": tf.FixedLenFeature([], dtype=tf.int64)
    }

    sequence_features = {
        "xyz_coords": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "material": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "size": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "rotation": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "pixel_coords": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "color": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "shape": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "question" : tf.FixedLenSequenceFeature([], dtype=tf.int64)
    }


    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        context_features=context_features,
        serialized=serialized_example,
        sequence_features=sequence_features
        )

    decoded_data = dict()
    for key in sequence_parsed:
        if key in ['xyz_coords', 'pixel_coords', 'rotation']:
            decoded_data[key] = tf.cast(tf.decode_raw(sequence_parsed[key], tf.float64), tf.float32, name=key)
        elif key in ['material', 'size', 'color', 'shape']:
            decoded_data[key] = tf.cast(tf.decode_raw(sequence_parsed[key], tf.int64), tf.int32, name=key)
        elif key in ['question']:
            decoded_data[key] = sequence_parsed[key]
        else:
            raise AttributeError

    return decoded_data, context_parsed


def inputs(data_type, obj_len_list, batch_size, num_epochs, num_threads=1):
    # filename = [os.path.join(data_dir, data_name) for data_name
    #             in data_name_list]

    filename = ['/home/jinwon/PycharmProjects/Relational_Network/code/processed_data/' + 'input_data_{}_len_{}.tfrecords'.format(data_type, len_key) for len_key in obj_len_list]
    filename_queue = tf.train.string_input_producer(filename, num_epochs)
    reader_sequence_output, reader_context_output = read_and_decode(filename_queue)

    # upg_no = reader_sequence_output['UPG_NO']
    # eitem_no = reader_sequence_output['E_ITEM_NO']
    # hw_key_en_nm = reader_sequence_output['HW_KEY_EN_NM_LIST']
    # hw_en_nm = reader_sequence_output['HW_EN_NM_LIST']
    # work_nm = reader_sequence_output['WORK_NM']
    #
    # hw_key_en_nm_len = reader_context_output['HW_KEY_EN_NM_LIST_LEN']
    # hw_en_nm_len = reader_context_output['HW_EN_NM_LIST_LEN']
    # work_nm_len = reader_context_output['WORK_NM_LEN']

    # reader_sequence_output['xyz_coords'].set_shape((num_object_pair, 6))
    # reader_sequence_output['pixel_coords'].set_shape((num_object_pair, 6))
    # reader_sequence_output['material'].set_shape((num_object_pair, 2))
    # reader_sequence_output['rotation'].set_shape((num_object_pair, 2))
    # reader_sequence_output['size'].set_shape((num_object_pair, 2))
    # reader_sequence_output['color'].set_shape((num_object_pair, 2))
    # reader_sequence_output['shape'].set_shape((num_object_pair, 2))

    # 'xyz_coords', 'material', 'size', 'rotation', 'pixel_coords', 'color', 'shape'
    # , reader_context_output['num_pair'],
    # reader_sequence_output['color'], reader_sequence_output['pixel_coords']
    batch = tf.train.batch([reader_sequence_output['xyz_coords'],
                            reader_sequence_output['pixel_coords'],
                            reader_sequence_output['rotation'],
                            reader_sequence_output['material'],
                            reader_sequence_output['color'],
                            reader_sequence_output['size'],
                            reader_sequence_output['shape'],
                            reader_sequence_output['question'],
                            reader_context_output['answer'],
                            reader_context_output['num_pair'],
                            reader_context_output['question_word_len']
                            ],
                             batch_size, dynamic_pad=True, allow_smaller_final_batch=False,
                           capacity = batch_size * 2, num_threads=num_threads)

    return batch


def prepare_data(len_key_list):
    def prepare_tfrecords(len_key_list, data_type):
        to_make_ilst = list()
        for len_key in len_key_list:
            filename = '/home/jinwon/PycharmProjects/Relational_Network/code/processed_data/input_data_{}_len_{}.tfrecords'.format(
                data_type, len_key)
            if not os.path.exists(filename):
                to_make_ilst.append(len_key)

        if not to_make_ilst:
            return
        else:
            if os.path.exists('processed_data/input_data_{}.pkl'.format(data_type)):
                print('loaded input_data_{}.pkl'.format(data_type))
                with open('processed_data/input_data_{}.pkl'.format(data_type), 'rb') as f:
                    data = pickle.load(f)
            else:
                print('processing input_data_{}.pkl'.format(data_type))
                data_path = '/home/jinwon/Downloads/CLEVR_v1.0/scenes/CLEVR_{}_scenes.json'.format(data_type)
                data, idx_to_value = read_and_preprocess_input_data(data_path)
                with open('processed_data/input_data_{}.pkl'.format(data_type), 'wb') as f:
                    pickle.dump(data, f)

                with open('processed_data/idx_to_value.pkl', 'wb') as f:
                    pickle.dump(idx_to_value, f)

            if os.path.exists('processed_data/question_data_{}.pkl'.format(data_type)):
                print('loaded question_data_{}.pkl'.format(data_type))
                with open('processed_data/question_data_{}.pkl'.format(data_type) ,'rb') as f:
                    question_data = pickle.load(f)
                with open('processed_data/answer_data_{}.pkl'.format(data_type) ,'rb') as f:
                    answer_data = pickle.load(f)
            else:
                print('processing question_data_{}.pkl'.format(data_type))
                data_path = '/home/jinwon/Downloads/CLEVR_v1.0/questions/CLEVR_{}_questions.json'.format(data_type)
                if data_type == 'train':
                    question_data, answer_data, word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word = read_and_preprocess_question_data(data_path)
                    with open('processed_data/question_answer_dict.pkl', 'wb') as f:
                        pickle.dump([word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word], f)
                else:
                    with open('processed_data/question_answer_dict.pkl', 'rb') as f:
                        convert_dict = pickle.load(f)
                    question_data, answer_data, word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word = read_and_preprocess_question_data(data_path, convert_dict)

                with open('processed_data/question_data_{}.pkl'.format(data_type), 'wb') as f:
                    pickle.dump(question_data, f)
                with open('processed_data/answer_data_{}.pkl'.format(data_type), 'wb') as f:
                    pickle.dump(answer_data, f)

            for len_key in to_make_ilst:
                filename = '/home/jinwon/PycharmProjects/Relational_Network/code/processed_data/input_data_{}_len_{}.tfrecords'.format(
                    data_type, len_key)
                writer = tf.python_io.TFRecordWriter(filename)
                for input_row, question_row, answer_row in zip(data[len_key], question_data, answer_data):
                    ex = make_example(**input_row, question=question_row, answer=answer_row)
                    writer.write(ex.SerializeToString())
                writer.close()
                print('tfrecord {} {} made'.format(data_type, len_key))

    prepare_tfrecords(len_key_list, 'train')
    prepare_tfrecords(len_key_list, 'val')


def run_test():
    len_key = 6
    num_pair = int(len_key * (len_key - 1) / 2)
    prepare_data(len_key)
    batch_size = 128
    num_epochs = 10
    with tf.Graph().as_default():
        data_name_list = [len_key]
        a = inputs(data_name_list, batch_size, num_epochs, num_pair)
        count = 0
        with tf.Session() as sess:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)
            try:
                while not coord.should_stop():
                    b = sess.run(a)
                    for elem in b:
                        print(elem.shape)
                    print('-------------------')
                    count += 1

            except tf.errors.OutOfRangeError:
                print('Done training --epoch limit reached')
                print(count)
            finally:
                coord.request_stop()
            coord.join(threads)


if __name__ == '__main__':
    run_test()
    # print('hello')
