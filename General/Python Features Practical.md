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
`**kwargs` : accept any number of keyword arguments, accessible via kwargs dict 
decomposing positional arguments, keyword arguments:
- `def add()`
- `add(*[a, b], **{'c': 'd', 'e': 'f'})`

## Global Interpreter Lock

basically, python interpreter is limited to single-threaded operation

## `if name=="__main__"`

only runs if the file was called directly

## Classes, instance methods, properties, instance variables, attributes

- cannot have collisions between instance variables and class methods
	- conventional fix is to have instance variable be prefixed with underscore

- instance variables
- instance methods (functions inside of classes)
	- instance methods, by convention, always have at least one argument, self
- `__` - dunder, double underscore
	- `__init__` : always run on class instantiation
	- `__str__` : string representation of class instantiation

## default case

- `_` e.g.
```python
match self.value:
	case _:
```
- matches case where `value = None`






## 🧭  Idea Compass
- West  (similar) 
[[Python Syntax Basics]]
- East (opposite)

- North (theme/question)

- South (what follows)

