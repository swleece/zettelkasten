---
date: 2023-12-22
time: 07:57
note_maturity: 🌱
tags:
---
# Python Features Practical

Patterns and features of python that I see in use and should understand.

## `self`

`self` is a convention used in object-oriented programming to refer to the instance of the class on which a method is being called
It is not a keyword in python, but rather a naming convention that is strongly adhered to.
Python passes `self` automatically.

## getters and setters

enables getters and setter class methods in python
python class will automatically recognize when getting or setting

### `@property` decorator (getter method)

- need to name the method exactly the same as the attribute
- always just have self in signature
used to turn a class method into a property of a class. this allows you to access the method like an attribute, without the need to call it as a function. It's a part of Python's approach to encapsulation, enabling you to implement getters and setters, thereby controlling access to private attributes in a class
```python
class MyClass:
    def __init__(self):
        self._my_attribute = 0

    @property
    def _my_attribute(self):
        return self._my_attribute

obj = MyClass()
print(obj._my_attribute)  # No need to use ()
```

### `@<my_attribute>.setter` decorator (setter method)

- always takes self and attribute name in signature 

## `@classmethod` decorator, Class methods

- don't have access to `self`
- 



## Mutable vs Immutable types

Immutable: str, int, float, bool, bytes, tuple
mutable: list, set, dict

When multiple variables are assigned to an immutable object, the object is actually copied.
When multiple variables are assigned to a mutable object, both variables point to same object.
(both are simply referencing the same object)

## Passing arguments

positional: must be provided first
- `add(a, b)
keyword: 
- `add(b = 1, a = 2)`
optional parameters: (z = None)
`*args` : accept any number of positional arguments, accessible via args tuple
`*options`: accept any number of positional arguments, accessible via args tuple
`**kwargs` : accept any number of keyword arguments, accessible via kwargs dict 
decomposing positional arguments, keyword arguments:
- `def add()`
- `add(*[a, b], **{'c': 'd', 'e': 'f'})`

## Global Interpreter Lock

basically, python interpreter is limited to single-threaded operation

## `if name=="__main__"`

only runs if the file was called directly

## Classes

**instance methods**: by convention, always have at least one argument, self
**class methods**: for operating class-level data (class attributes) or when the method's behavior is related to the class itself, not to specific instances.
**static methods**: `@staticmethod`
**properties**
**instance variables**
**attributes**: class level data

- cannot have collisions between instance variables and class methods
	- conventional fix is to have instance variable be prefixed with underscore

- `__` - dunder, double underscore
	- `__init__` : always run on class instantiation
	- `__str__` : string representation of class instantiation

### Example Python Class

```Python
class Shape:
    num_shapes = 0  # Class variable

    def __init__(self, name, color):
        self.name = name # Instance variable
        self.color = color
        Shape.num_shapes += 1

    def describe(self):  # Instance method
        return f"This is a {self.color} {self.name}."

    @staticmethod
    def validate_color(color):  # Static method
        valid_colors = ['red', 'green', 'blue', 'yellow']
        return color.lower() in valid_colors

    @classmethod
    def create_square(cls, side_length, color):  # Class method
        if not cls.validate_color(color):
            raise ValueError("Invalid color")
        return cls("square", color)

    @classmethod
    def get_num_shapes(cls):  # Class method
        return cls.num_shapes

# Usage
red_circle = Shape("circle", "red")
print(red_circle.describe())  # Instance method

print(Shape.validate_color("green"))  # Static method
print(Shape.validate_color("purple"))  # Static method

blue_square = Shape.create_square(5, "blue")  # Class method
print(blue_square.describe())  # Instance method

print(Shape.get_num_shapes())  # Class method
```

# Usage
red_circle = Shape("circle", "red")
print(red_circle.describe())  # Instance method

print(Shape.validate_color("green"))  # Static method
print(Shape.validate_color("purple"))  # Static method

blue_square = Shape.create_square(5, "blue")  # Class method
print(blue_square.describe())  # Instance method

print(Shape.get_num_shapes())  # Class method

## default case
- `_` e.g.
```python
match self.value:
	case _:
```
- matches case where `value = None`

### [**Descriptors**](https://docs.python.org/3/howto/descriptor.html): 
- enable objects to customize attribute lookup, storage, and deletion
- interesting descriptors typically run computations on demand instead of storing them with a class instance
- enable dynamic lookups
- A popular use for descriptors is managing access to instance data
- A [descriptor](https://docs.python.org/3/glossary.html#term-descriptor) is what we call any object that defines `__get__()`, `__set__()`, or `__delete__()`.
- Optionally, descriptors can have a `__set_name__()` method. This is only used in cases where a descriptor needs to know either the class where it was created or the name of class variable it was assigned to. (This method, if present, is called even if the class is not a descriptor.)
- Descriptors get invoked by the dot operator during attribute lookup. If a descriptor is accessed indirectly with `vars(some_class)[descriptor_name]`, the descriptor instance is returned without invoking it.
- Descriptors only work when used as class variables. When put in instances, they have no effect.
- The main motivation for descriptors is to provide a hook allowing objects stored in class variables to control what happens during attribute lookup.
- Traditionally, the calling class controls what happens during lookup. Descriptors invert that relationship and allow the data being looked-up to have a say in the matter.

### **Abstract Base Class (ABC)**
Abstract base classes complement [duck-typing](https://docs.python.org/3/glossary.html#term-duck-typing) by providing a way to define interfaces when other techniques like [`hasattr()`](https://docs.python.org/3/library/functions.html#hasattr "hasattr") would be clumsy or subtly wrong (for example with [magic methods](https://docs.python.org/3/reference/datamodel.html#special-lookup)).
ABCs introduce virtual subclasses, which are classes that don’t inherit from a class but are still recognized by [`isinstance()`](https://docs.python.org/3/library/functions.html#isinstance "isinstance") and [`issubclass()`](https://docs.python.org/3/library/functions.html#issubclass "issubclass"); see the [`abc`](https://docs.python.org/3/library/abc.html#module-abc "abc: Abstract base classes according to :pep:`3119`.") module documentation. 
Python comes with many built-in ABCs for data structures (in the [`collections.abc`](https://docs.python.org/3/library/collections.abc.html#module-collections.abc "collections.abc: Abstract base classes for containers") module), numbers (in the [`numbers`](https://docs.python.org/3/library/numbers.html#module-numbers "numbers: Numeric abstract base classes (Complex, Real, Integral, etc.).") module), streams (in the [`io`](https://docs.python.org/3/library/io.html#module-io "io: Core tools for working with streams.") module), import finders and loaders (in the [`importlib.abc`](https://docs.python.org/3/library/importlib.html#module-importlib.abc "importlib.abc: Abstract base classes related to import") module). You can create your own ABCs with the [`abc`](https://docs.python.org/3/library/abc.html#module-abc "abc: Abstract base classes according to :pep:`3119`.") module.

```Python
# descriptors, abstract base classes
from abc import ABC, abstractmethod

class Validator(ABC): # abstract base class, descriptor
    def __set_name__(self, owner, name):
        self.private_name = '_' + name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(self, obj, value):
        self.validate(value)
        setattr(obj, self.private_name, value)

    @abstractmethod
    def validate(self, value):
        pass

class OneOf(Validator): # `OneOf` verifies that a value is one of a restricted set of options.
	# t = OneOf('a')
	# t.validate('b')
    def __init__(self, *options):
        self.options = set(options)

    def validate(self, value):
        if value not in self.options:
            raise ValueError(f'Expected {value!r} to be one of {self.options!r}')
```


### Python Attribute lookup process
When you access an attribute using the dot notation (like `a.x`), Python follows a specific lookup order: a. It first checks the instance dictionary. b. If not found, it then looks in the class dictionary. c. If still not found, it checks any base classes (following the method resolution order).



## 🧭  Idea Compass
- West  (similar) 
[[Python Syntax Basics]]
- East (opposite)

- North (theme/question)

- South (what follows)

