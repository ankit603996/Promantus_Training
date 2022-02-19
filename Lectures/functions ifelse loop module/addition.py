"""This module has a function for addition
"""


def add_two_number(a, b):
    """
    :param a: input parameter
    :param b: output parameter
    :return: a+b as an output
    """
    print("Calling add two number")
    return a + b


def add_list(*args):
    """
    :param args: get the list of number
    :return: the sum of the list of the numbers as an output
    """
    print("calling add_list function")
    sum_list = 0
    for i in args:
        sum_list += i
    return sum_list

