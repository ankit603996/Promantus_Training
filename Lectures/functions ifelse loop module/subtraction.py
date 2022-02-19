"""
This module has functions for subtraction
"""


def sub_two_number(a, b):
    """
    :param a: Numeric input
    :param b: Numeric input
    :return: sum of the given inputs
    """
    print("Calling sub_two_number function")
    return a - b


def sub_list_numbers(*args):
    """
    :param args: list of numbers
    :return:
    """
    print("Calling list of given numbers")
    sub_list = 0
    for i in args:
        sub_list -= i
    return sub_list


def sub_three_number(x, y, z):
    """

    :param x: One of the inputs to func
    :param y: One of the inputs to func
    :param z: One of the inputs to func
    :return: Gives the subtracted values from given inputs
    """
    list_input = [x, y, z]
    sub_value = 0
    for i in list_input:
        sub_value -= i
    return sub_value
