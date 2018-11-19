import cv2
import os
import numpy as np
import random
# import cPickle as pickle
import pickle

from skimage.draw import circle
from skimage.draw import rectangle
from skimage.draw import polygon

train_size = 9800
test_size = 200
img_size = 128
size = 4
slack = 3

num_shape = 2
num_rel_qst = 5 #9
num_nonrel_qst = 3
question_size = 11 + (num_rel_qst - 3)  ##6 for one-hot vector of color, 2 for question
                      # type,

answer_size_before_color = 10 + (num_shape - 2)  # 0 ~ 9 answer_dict
answer_size_before_count = 4  + (num_shape - 2) # 0 ~ 4


# 3 for question subtype
"""Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""

nb_questions = 10



colors = [
    (255, 0, 0),  ##r
    (0, 255, 0),  ##g
    (0, 0, 255),  ##b
    (255, 156, 0),  ##orange
    (148, 0, 211),  ##darkviolet
    (255, 255, 0)  ##y
]


color_dict = {
        0: 'r',
        1: 'g',
        2: 'b',
        3: 'o',
        4: 'v',
        5: 'y',
    }


question_type_dict = {
            0: 's',
            1: 'h',
            2: 'v',
            3: 'cl_c',
            4: 'f_c',
            5: 'co',
            6: 'cl_s',
            7: 'f_s',
            8:'f_cl_c',
            9:'cl_f_c'
        }



# answer_dict = {
#             0: 'y',
#             1: 'n',
#             2: 'rec',
#             3: 'cir',
#             4: 'tri',
#             5: '1',
#             6: '2',
#             7: '3',
#             8: '4',
#             9: '5',
#             10: '6',
#             11: 'r',
#             12: 'g',
#             13: 'b',
#             14: 'o',
#             15: 'v',
#             16: 'y'
#         }

answer_dict = {
            0: 'y',
            1: 'n',
            2: 'rec',
            3: 'cir',
            4: '1',
            5: '2',
            6: '3',
            7: '4',
            8: '5',
            9: '6',
            10: 'r',
            11: 'g',
            12: 'b',
            13: 'o',
            14: 'v',
            15: 'y'
        }




def draw_triangle(img, img_size, x, y, size, color):
    # img = np.zeros((img_size, img_size, 3))
    size = size * 1.5
    upper = np.array((0, size))
    spin_mat = np.array(
        [np.cos(np.pi / 3 * 2), -np.sin(np.pi / 3 * 2), np.sin(np.pi / 3 * 2),
         np.cos(np.pi / 3 * 2)]).reshape(2, 2)
    lower_right = np.matmul(spin_mat, upper)
    lower_left = np.matmul(spin_mat, lower_right)

    vertices = np.stack((upper, lower_right, lower_left), axis=1).reshape(-1, 3)
    vertices[0, :] = vertices[0, :] + x
    vertices[1, :] = vertices[1, :] + y

    tmp = vertices[0, :].copy()
    vertices[0, :] = vertices[1, :]
    vertices[1, :] = tmp

    rr, cc = polygon(vertices[0, :], vertices[1, :])

    img[rr, cc, 0] = color[0]
    img[rr, cc, 1] = color[1]
    img[rr, cc, 2] = color[2]

    #     print(rr)
    #     print(cc)
    return img



def center_generate(objects):

    while True:
        cnt = 0
        while True:
            pas = True

            center = np.random.randint(0 + size + slack, img_size - size - slack, 2)
            if len(objects) > 0:
                for name, c, shape in objects:
                    if ((center - c) ** 2).sum() < (1 * (size * 2) ** 2):
                        pas = False
                        cnt +=1

            if cnt > 2000 and not pas:
                print('broke reset')
                break

            if pas:
                return center


def build_dataset_all_question():
    objects = []
    img = np.ones((img_size, img_size, 3)) * 255
    for color_id, color in enumerate(colors):
        center = center_generate(objects)
        shape = np.random.randint(num_shape)
        if shape == 0:
            start = (center[0] - size, center[1] - size)
            end = (center[0] + size, center[1] + size)
            # cv2.rectangle(img, start, end, color, -1)
            rr, cc = rectangle(start, end)
            img[rr, cc] = color
            objects.append((color_id, center, 'rec'))

        elif shape == 1:
            center_ = (center[0], center[1])
            # cv2.circle(img, center_, size, color, -1)
            rr, cc = circle(*center_, size + 1)
            img[rr, cc] = color

            objects.append((color_id, center, 'cir'))

        # elif shape == 2:
        #     center_ = (center[1] , center[0])
        #     img = draw_triangle(img, img_size, *center_, size  , color)
        #     objects.append((color_id, center, 'tri'))



    rel_questions = []
    norel_questions = []
    rel_answers = []
    norel_answers = []
    # """Non-relational questions"""
    for color in color_dict.keys():
        for subtype in range(num_nonrel_qst):
            question = np.zeros((question_size))
            # color = random.randint(0, 5)
            question[color] = 1
            question[6] = 1
            # subtype = random.randint(0, 2)
            question[subtype + 8] = 1
            norel_questions.append(question)
            """Answer : [yes, no, rectangle, circle, r, g, b, o, k, y]"""
            if subtype == 0:
                """query shape->rectangle/circle"""
                if objects[color][2] == 'rec':
                    answer = 2
                elif objects[color][2] == 'cir':
                    answer = 3
                # elif objects[color][2] == 'tri':
                #     answer = 4
                else:
                    print('error in dat')
                    exit()

            elif subtype == 1:
                """query horizontal position->yes/no"""
                if objects[color][1][0] < img_size / 2:
                    answer = 0
                else:
                    answer = 1

            elif subtype == 2:
                """query vertical position->yes/no"""
                if objects[color][1][1] < img_size / 2:
                    answer = 0
                else:
                    answer = 1
            norel_answers.append(answer)

    """Relational questions"""
    for color in color_dict.keys():
        for subtype in range(num_rel_qst):
            question = np.zeros((question_size))
            # color = random.randint(0, 5)
            question[color] = 1
            question[7] = 1
            # subtype = random.randint(0, 2)
            question[subtype + 8] = 1
            rel_questions.append(question)

            if subtype == 0:
                """closest-to->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                dist_list[dist_list.index(0)] = (img_size ** 2) * 2 #max distance
                closest = dist_list.index(min(dist_list))


                answer = objects[closest][0] + answer_size_before_color

            elif subtype == 1:
                """farthest-from->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                farthest = dist_list.index(max(dist_list))



                answer = objects[farthest][0] + answer_size_before_color

            elif subtype == 2:
                """count->1~6"""
                my_obj = objects[color][2]
                count = -1
                for obj in objects:
                    if obj[2] == my_obj:
                        count += 1

                answer = count + answer_size_before_count


            elif subtype == 3:
                """closest-to->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                dist_list[dist_list.index(0)] = (img_size ** 2) * 2  # max distance
                closest = dist_list.index(min(dist_list))

                if objects[closest][2] == 'rec':
                    answer = 2
                elif objects[closest][2] == 'cir':
                    answer = 3
                # elif objects[closest][2] == 'tri':
                #     answer = 4
                else:
                    print('error in data')
                    exit()


            elif subtype == 4:
                """farthest-from->rectangle/circle"""
                my_obj = objects[color][1]
                dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
                farthest = dist_list.index(max(dist_list))

                if objects[farthest][2] == 'rec':
                    answer = 2
                elif objects[farthest][2] == 'cir':
                    answer = 3
                # elif objects[farthest][2] == 'tri':
                #     answer = 4
                else:
                    print('error in data')
                    exit()

            # elif subtype == 5:
            #     """farthest-from-closest"""
            #     my_obj = objects[color][1]
            #     dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            #     farthest = dist_list.index(max(dist_list))
            #
            #     farthest_obj = objects[farthest][1]
            #     dist_list = [((farthest_obj - obj[1]) ** 2).sum() for obj in objects]
            #     dist_list[dist_list.index(0)] = (img_size ** 2) * 2  # max distance
            #     farthest_closest = dist_list.index(min(dist_list))
            #
            #     answer = objects[farthest_closest][0] + answer_size_before_color
            #
            # elif subtype == 6:
            #     """closest-from-farthest"""
            #     my_obj = objects[color][1]
            #     dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            #     dist_list[dist_list.index(0)] = (img_size ** 2) * 2  # max distance
            #     closest = dist_list.index(min(dist_list))
            #
            #     closest_obj = objects[closest][1]
            #     dist_list = [((closest_obj - obj[1]) ** 2).sum() for obj in objects]
            #     closest_farthest = dist_list.index(max(dist_list))
            #
            #     answer = objects[closest_farthest][0] + answer_size_before_color
            #
            # elif subtype == 7:
            #     """farthest-from-closest"""
            #     my_obj = objects[color][1]
            #     dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            #     farthest = dist_list.index(max(dist_list))
            #
            #     farthest_obj = objects[farthest][1]
            #     dist_list = [((farthest_obj - obj[1]) ** 2).sum() for obj in objects]
            #     dist_list[dist_list.index(0)] = (img_size ** 2) * 2  # max distance
            #     farthest_closest = dist_list.index(min(dist_list))
            #
            #     if objects[farthest_closest][2] == 'rec':
            #         answer = 2
            #     elif objects[farthest_closest][2] == 'cir':
            #         answer = 3
            #
            # elif subtype == 8:
            #     """closest-from-farthest"""
            #     my_obj = objects[color][1]
            #     dist_list = [((my_obj - obj[1]) ** 2).sum() for obj in objects]
            #     dist_list[dist_list.index(0)] = (img_size ** 2) * 2  # max distance
            #     closest = dist_list.index(min(dist_list))
            #
            #     closest_obj = objects[closest][1]
            #     dist_list = [((closest_obj - obj[1]) ** 2).sum() for obj in objects]
            #     closest_farthest = dist_list.index(max(dist_list))
            #
            #     if objects[closest_farthest][2] == 'rec':
            #         answer = 2
            #     elif objects[closest_farthest][2] == 'cir':
            #         answer = 3
                








            rel_answers.append(answer)

    relations = (rel_questions, rel_answers)
    norelations = (norel_questions, norel_answers)

    # img = img / 255.
    dataset = (img, relations, norelations)
    return dataset


def generate_data(data_option=None):
    if data_option:
        dirs = 'data/Sort-of-CLEVR/raw_data/{}'.format(data_option)
    else:
        dirs = 'data/Sort-of-CLEVR/raw_data'

    try:
        os.makedirs(dirs)
    except:
        print('directory {} already exists'.format(dirs))

    filename = os.path.join(dirs, 'sort-of-clevr.pickle')

    if not os.path.exists(filename):

        # print('building test datasets...')
        # test_datasets = [build_dataset() for _ in range(test_size)]
        # print('building train datasets...')
        # train_datasets = [build_dataset() for _ in range(train_size)]

        print('building test datasets...')
        test_datasets = [build_dataset_all_question() for _ in range(test_size)]
        print('building train datasets...')
        train_datasets = [build_dataset_all_question() for _ in range(train_size)]

        print('saving datasets...')

        with  open(filename, 'wb') as f:
            pickle.dump((train_datasets, test_datasets), f)

        with open(os.path.join(dirs, 'ans_color_qst_dict.pickle'), 'wb') as f:
            pickle.dump((answer_dict, color_dict, question_type_dict), f)

        print('datasets saved at {}'.format(filename))


def test():
    a = [build_dataset_all_question() for _ in range(10)]
    import matplotlib.pyplot as plt
    for val in a:
        img, relations, nonrelations = val
        (rel_questions, rel_answers) = relations
        (norel_questions, norel_answers) = nonrelations
        for q, a in zip(rel_questions, rel_answers):
            for subtype in [5, 6]:
                if q[subtype + 8] == 1:
                    color = list(q[:6]).index(1)
                    print(question_type_dict[subtype + 3], color_dict[color],
                          answer_dict[a])
                    plt.imshow(img)
                    plt.show()







if __name__ == '__main__':
    test()
