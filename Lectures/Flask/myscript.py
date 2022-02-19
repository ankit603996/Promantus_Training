# -*- coding: utf-8 -*-
"""
Created on Mon Aug 23 00:41:13 2021

@author: ankit
"""


# Python program to execute
# function directly
def my_function():
    print ("I am inside function")
    print(__name__)
def my_function2():
    return __name__

# We can test function by calling it.
my_function()