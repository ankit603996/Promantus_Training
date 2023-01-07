'''
The id() Function : built-in function
get the address of an object from the RAM
'''

p = 'luck'
print(id('luck'))
print(id(p))



'''
Name Space

A Python namespace is an efficient system where each and every object in Python has a unique name. 
Every package, module, class, function and method in Python occupies a namespace in which variable names are set. 

Python namespace is a collection or a system of names that ensures all
the value names in a program are unique, and we can use them without any conflict.

Namespaces are collections of different objects that are associated with unique names
whose lifespan depends on the scope of a variable. The scope is a region from where we
can access a particular object.

There are three levels of scopes: built-in (outermost), global, and local.
We can only access an object if its scope allows us to do so.

Types of Python Namespace
Built-in namespace – A Python namespace gets created when we start using the interpreter, and it exists as long as it is in use. When the Python interpreter runs without any user-defined modules, methods, etc., built-in functions like print() and id() are always present and available from any part of a program. These functions exist in the built-in namespace.

Global namespace – A global namespace gets created when you create a module.

Local namespace – This namespace gets created when you create new local functions. 
'''

# varA in the global namespace
varA = 25
def A_func():
    # varB in the local namespace
    varB = 83
    def B_inner_func():
        # varC in the nested local namespace
        varC = 9




# name space and variable scope

#  LEGB Rules
# Closures
y = 10
def outer():
    def inner():
        y=6
        y = y+1
        x = 3
        print("output of inner()",x,y)
        return "hello"
    return inner
print("outer y " , y)
a= outer()
print(a)
print(hex(id(a)))



## decorator

# defining a decorator
def hello_decorator(func):
    # inner1 is a Wrapper function in which the argument is called
    # inner function can access the outer local functions like in this case "func"
    def inner1():
        print("Hello, this is before function execution")
        # calling the actual function now inside the wrapper function.
        func()
        print("This is after function execution")
    return inner1

# defining a function, to be called inside wrapper
def function_to_be_used():
    print("This is inside the function !!")

# passing 'function_to_be_used' inside the decorator to control its behaviour
function_to_be_used = hello_decorator(function_to_be_used)
# calling the function
function_to_be_used()

##### We can also write it as  ######
@hello_decorator # decorator function
def function_to_be_used():
    print("This is inside the function !!")
function_to_be_used()


##### Decorator
class Cellphone:
    def __init__(self, brand, number):
        self.brand = brand
        self.number = number
    def get_number(self):
        return self.number
    @staticmethod
    def get_emergency_number():
        return "911"
Cellphone.get_emergency_number()
# '911'
#get_number is a regular instance method of the class
# and requires the creation of an instance
#get_emergency_number is a static method because it
#is decorated with the @staticmethod decorator


#### classmethod
class Date(object):

    def __init__(self, day=0, month=0, year=0):
        self.day = day
        self.month = month
        self.year = year

    @classmethod
    def from_string(cls, date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        date1 = cls(day, month, year)
        return date1

    @staticmethod
    def is_date_valid(date_as_string):
        day, month, year = map(int, date_as_string.split('-'))
        return day <= 31 and month <= 12 and year <= 3999

date2 = Date.from_string('11-09-2012')
is_date = Date.is_date_valid('11-09-2012')

