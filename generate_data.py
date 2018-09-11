import pickle
import tensorflow as tf
import os, json, re, itertools, collections
import numpy as np
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



def read_and_preprocess_scene_data(data_path):
    scene = json.loads(open(data_path).read())
    data = list()

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
        img_idx = element['image_index']
        row = collections.defaultdict(list)
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
        data.append((row, img_idx))

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


def inputs(data_type, batch_size, num_epochs, num_threads=10):
    filename = ['processed_data/input_data_{}.tfrecords'.format(data_type)]
    filename_queue = tf.train.string_input_producer(filename, num_epochs)
    reader_sequence_output, reader_context_output = read_and_decode(filename_queue)

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
                           capacity = batch_size * 10, num_threads=num_threads)

    # dataset = tf.data.Dataset.list_files(filename)
    # dataset = dataset.map(read_and_decode, num_parallel_calls=1)
    # dataset = dataset.prefetch(batch_size * 100)
    # dataset = dataset.shuffle(buffer_size= 1000)
    # dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # dataset = dataset.repeat(num_epochs)
    #
    # iterator = dataset.make_one_shot_iterator()
    # batch = iterator.get_next()
    return batch

def prepare_data():
    def prepare_tfrecords(data_type):

        filename = 'processed_data/input_data_{}.tfrecords'.format(data_type)

        if os.path.exists(filename):
            print('{} exists'.format(filename))
            return
        else:
            if not os.path.exists('processed_data'):
                os.makedirs('processed_data')

            pickled_input_data = 'processed_data/input_data_{}.pkl'.format(data_type)
            if os.path.exists(pickled_input_data):
                print('loaded input_data_{}.pkl'.format(data_type))
                with open(pickled_input_data, 'rb') as f:
                    data = pickle.load(f)
            else:
                print('processing input_data_{}.pkl'.format(data_type))
                data_path = 'data/CLEVR_v1.0/scenes/CLEVR_{}_scenes.json'.format(data_type)
                data, idx_to_value = read_and_preprocess_scene_data(data_path)
                with open(pickled_input_data, 'wb') as f:
                    pickle.dump(data, f)

                with open('processed_data/idx_to_value.pkl', 'wb') as f:
                    pickle.dump(idx_to_value, f)

            if os.path.exists('processed_data/question_data_{}.pkl'.format(data_type)):
                print('loaded qa_data_{}.pkl'.format(data_type))
                # with open('processed_data/question_data_{}.pkl'.format(data_type) ,'rb') as f:
                #     question_data = pickle.load(f)
                # with open('processed_data/answer_data_{}.pkl'.format(data_type) ,'rb') as f:
                #     answer_data = pickle.load(f)

                with open('processed_data/qa_data_{}.pkl'.format(data_type), 'wb') as f:
                    qa_data = pickle.load(f)
            else:
                print('processing question_data_{}.pkl'.format(data_type))
                data_path = 'data/CLEVR_v1.0/questions/CLEVR_{}_questions.json'.format(data_type)
                if data_type == 'train':
                    qa_data, word_to_idx, idx_to_word, answer_word_to_idx, \
                    answer_idx_to_word = read_and_preprocess_question_data(data_path)
                    with open('processed_data/question_answer_dict.pkl', 'wb') as f:
                        pickle.dump([word_to_idx, idx_to_word, answer_word_to_idx, answer_idx_to_word], f)
                else:
                    with open('processed_data/question_answer_dict.pkl', 'rb') as f:
                        convert_dict = pickle.load(f)
                    qa_data, word_to_idx, idx_to_word, answer_word_to_idx, \
                    answer_idx_to_word = read_and_preprocess_question_data(data_path, convert_dict)

                with open('processed_data/qa_data_{}.pkl'.format(data_type), 'wb') as f:
                    pickle.dump(qa_data, f)

            filename = 'processed_data/input_data_{}.tfrecords'.format(
                data_type)
            writer = tf.python_io.TFRecordWriter(filename)

            for input_row in data:
                input_data, img_idx = input_row
                for question, answer in qa_data[img_idx]:
                    ex = make_example(**input_data, question=question, answer=answer)
                    writer.write(ex.SerializeToString())
            writer.close()
            print('tfrecord {} made'.format(data_type))

    prepare_tfrecords('train')
    prepare_tfrecords('val')

