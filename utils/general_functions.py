import json


def load_dict(file_path):
    """

    :param file_path:
    :return:
    """
    in_file = open(file_path, 'r')
    return json.load(in_file)


def save_dict(file_path, dictionary):
    """

    :param file_path:
    :param dictionary:
    :return:
    """
    out_file = open(file_path, 'w+')
    json.dump(dictionary, out_file)
