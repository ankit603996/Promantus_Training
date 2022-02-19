import modules
from modules.addition import add_two_number, add_list
from modules.subtraction import sub_two_number, sub_list_numbers

print(add_two_number(1, 2))
print(add_list(1, 2, 3, 4))

print(sub_two_number(1, 2))
print(sub_list_numbers(10, 9, 8))

print(modules.__name__)
