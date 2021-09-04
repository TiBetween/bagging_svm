import cv2 as cv


def unpickle(file):
    """
    数据集提供的解包函数
    :param file:batch文件路径
    :return: batch中以字典保存的数据集
    """
    import pickle
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def load_dataset_with_full_pic(file):
    """
    返回以完整图片为格式的数据集拆分结果
    :param file: batch文件路径
    :return:
        pic_list        :完整图片文件列表
        data_label      :对应标签
        data_filenames  :对应文件名
    """
    dict_ = unpickle(file)
    _temp_pic = dict_[list(dict_.keys())[2]]
    pic_list = []
    for i in range(10000):
        temp_pic_r = _temp_pic[i][:1024].reshape(32, 32)
        temp_pic_g = _temp_pic[i][1024:2048].reshape(32, 32)
        temp_pic_b = _temp_pic[i][2048:3072].reshape(32, 32)
        temp_pic = cv.merge([temp_pic_r, temp_pic_g, temp_pic_b])
        pic_list.append(temp_pic)

    data_label = dict_[list(dict_.keys())[1]]
    data_filenames = dict_[list(dict_.keys())[3]]
    return pic_list, data_label, data_filenames


def load_dataset_with_pic_tune(file):
    """
    返回以图片三通道为格式的数据集拆分结果
    :param file: batch文件路径
    :return:
        pic_list_r          :红色通道
        pic_list_g          :绿色通道
        pic_list_b          :蓝色通道
        data_label          :对应标签
        data_filenames      :对应文件名
    """
    dict_ = unpickle(file)
    _temp_pic = dict_[list(dict_.keys())[2]]
    pic_list_r = []
    pic_list_g = []
    pic_list_b = []
    for i in range(10000):
        pic_list_r.append(_temp_pic[i][:1024].reshape(32, 32))
        pic_list_g.append(_temp_pic[i][1024:2048].reshape(32, 32))
        pic_list_b.append(_temp_pic[i][2048:3072].reshape(32, 32))

    data_label = dict_[list(dict_.keys())[1]]
    data_filenames = dict_[list(dict_.keys())[3]]
    return pic_list_r, pic_list_g, pic_list_b, data_label, data_filenames


def b2s(b):
    return str(b, encoding="utf-8")


def s2b(s):
    return bytes(s, encoding="utf-8")


def load_label_dict(file):
    """
    返回标签序号和对应文字标签的字典
    :param file: 文件路径
    :return: 返回标签序号和对应文字标签的字典
    """
    dict_ = unpickle(file)
    index = 0
    label_list = dict_[s2b("label_names")]

    dict_ = {}
    for b in label_list:
        dict_[index] = b2s(b)
        index += 1
    return dict_


if __name__ == '__main__':
    # pics_list, pics_label, pic_filename = load_dataset_with_full_pic("./cifar-10-batches-py/data_batch_1")
    print(load_label_dict("./cifar-10-batches-py/batches.meta"))
