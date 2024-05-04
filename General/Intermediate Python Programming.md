---
date: 2024-01-15
time: 09:12
note_maturity: 🌱
tags:
  - youtube
link: https://www.youtube.com/watch?v=HGOBQPFzWKo
---
# Intermediate Python Programming Course - youtube

## Lists 

- a collection data type that is **ordered**, **mutable**, and **allows duplicate elements**
- `[1, 2, 3]`

## Tuples

- a collection data type that is **ordered** and **immutable**
- used for objects that belong together

## Dictionaries 

- a collection data type that is **unordered** and **mutable**, consisting of a **collection of key-value pairs**

## Sets

- a collection data type that is **unordered**, **mutable**, and does **not allow duplicate elements**
- `myset = {1, 2, 3}`

## Strings 

- a collection data type that is ordered, immutable, and is used for text representation
- formatting:
	- `%`, `.format()`, f-strings
- 

## Collections 

- module that implements special container data types and provides alternatives with additional functionality
- Examples include: `Counter`, `namedtuple`, `OrderedDict`, `defaultdict`, `deque`
- `Counter`: container that stores the elements as dictionary keys and their counts as dictionary values
	- `a = 'aaabbc'
	- `my_counter = Counter(a) # Counter({'a': 5, 'b': 2, 'c': 1})
- 

## Itertools 

- a collection of tools for handling iterators (data types that can be used for a for loop (list etc.))
- Examples include: `product`, `permutations`, `combinations`, `accumulate`, `groupby`, `infinite` iterators

## Lambda Functions 

- a small one line anonymous function defined without a name
- `lambda arguments: expression`
- `add10 = lambda x: x + 10`
- `add10(5) # 15)`
- can have multiple arguments
- typically used when you just need something simple to be done once
- or as an argument for higher order functions
-  Examples include: ``

## Exceptions and Errors

- a python program terminates as soon as it encounters an error
	- error can be either a syntax error or an exception
- Syntax errors:
	- when parser detects a syntactically invalid statement
- built in Exception error examples:
	- type errors (adding a string and int)
	- import exceptions (module not found error)
	- name error (name 'c' is not defined)
	- file not found error
- `Raise` keyword
	- enables you to force an exception to be raised in a given condition
- `assert ()`
	- will throw an assertion error if condition not true
	- `assert (x >= 0), 'x is not positive`
- to handle exceptions, use `try` `catch` block
	- `try:`
		- `a = 5 / 0`
	- `except ZeroDivisionError as e:
		- `print(e)
	- `except TypeError as e:
		- `print(e)
	- `else:
		- `print('everything fine')
	- `finally: # used for cleanup operations
		- `print('cleaning up')`
	- good practice to specify the type of exception you want to catch
- can define your own exceptions
- `class ValueTooHighError(Exception) # using Exception base class`
	- `pass
- `def test_value(x):
	- `if x > 100:
		- `raise ValueTooHighError('value is too high')`

## Logging

- use logging module to log to `debug, info, warning, error, critical`
	- by default only message with level warning and above are printed
- can use the `traceback` module to format errors with traceback
- Rotating file handlers: (`from logging.handlers import RotatingFileHandler`)
	- used to keep track of most recent events
	- enables you to log to log files with set log file size
	- also timed rotating file handler for time based log writing

## JSON

- built in json module useful for encoding/serialization and decoding/deserialization json data
- `import json`
- `personJSON = json.dumps(person) # convert to JSON, can specify indent, separators, sort_keys`
- `with open('person.json', 'w') as file: # write to a file
	- `json.dump(person, file, indent=4)
- `person = json.loads(personJSON) # convert back to python dictionary object`

## Random Numbers 

- random module used to generate pseudo-random numbers (are actually reproducible)
	- numbers can be reproduced using a seed
- secrets module for generating cryptographically (methods: `randbelow`, `randbits`, `choice`)
	- will generate a truly random number, not as fast
- numpy module for generating arrays 
	- `a = np.random.rand(3) # produces 1d array with three random floats`

## Decorators 

- two types, Function Decorators and Class Decorators
- a function that takes another function as an argument and extends the behavior of the function without modifying it
- functions in Python are first class objects (can be defined inside another function, passed as argument, or returned from function)
- functools
	- preserves the info the function being used
- Example template:
```python
import functools

def my_decorator(func):

	@functools.wrap(func)
	def wrapper(*args, **kwargs)
		# do something before...
		result = func(*args, **kwargs)
		# do something after
		return result
	return wrapper

@start_end_decorator
def add5(x):
	return x + 5
```
- if multiple decorators are applied, they will be executed in the order provided
- Class Decorators: typically used to maintain/update state
	- use cases: timer to calc execution time, debug decorator, cache return values etc.

## Generators 

- functions that lazily return an object that can be iterated over
- objects are generated one at a time
- are very memory efficient, good for working with large datasets
```python
def mygenerator():
	yield 1
	yield 2
	yield 3

g = mygenerator()

for i in g:
	print(i) # will print 1 then 2 then 3

value = next(g) # allows iteration through yielded values one at a time
```
- can be used as input to functions that take in iterables
- enables you to attach state to function execution
- fibonacci sequence is a typical use case
- generator expressions:
	- like list comprehensions but with parentheses instead of square brackets
	- `mygenerator = (i for i in range(10) if i % 2 == 0) # evens`
- use `list()` to convert a generator to list

## Threading vs Multiprocessing 

A **process** is an instance of a program (e.g. a [[Python interpreter]])
Pros:
- takes advantage of multiple cpu's and cores
- separate memory space -> memory is not shared between processes
- great for cpu-bound processing
- new process is started independently from other processes
- processes are interruptable/killable
- one global interpreter lock (GIL) for each process -> avoids GIL limitation
Cons:
- heavyweight
- starting a process is slower than starting a thread
- more memory
- inter-process communication (IPC) is more complicated
A **thread** is an entity within a process that can be scheduled for execution. A process can spawn multiple threads.
Pros:
- all threads within a process share the same memory
- lightweight
- starting a thread is faster than starting a process
- good for I/O-bound tasks (talking to slow devices or network)
Cons:
- threading is limited by the GIL: only one thread at a time
- no effect for cpu-bound tasks
- not interruptable/killable
- careful with race conditions
GIL: **[[global interpreter lock]]**
- a lock that allows only one thread at a time to execute in python
- needed in CPython because memory management is not thread-safe
- due to reference counting, keeping track of number of references to an object 
- to avoid the GIL:
	- use multiprocessing
	- use a different, free-threaded Python implementation (Jython, IronPython)
	- use Python as a wrapper for third-party libraries that are wrappers to execute code in C (numpy, scipy)

## Multithreading

- use multithreading module
- data can be shared between threads since they live in the same memory space

## Multiprocessing 

- use multiprocessing module
- use `num_processes = os.cpu_count()`
- similar to multithreading
- processes do not live in the same memory space, need to use `Value` or `Array` from multiprocessing module

## Function Arguments

- Arguments vs Parameters
- Parameters are the variables that are used inside parentheses when defining a function
- Arguments are the actual values used when calling the function
- positional vs keyword arguments
- generally better to use keyword arguments for readability
	- you can also force arguments to be keyword args by providing them after  a `*` / `*args`
	- e.g. `def foo(*args, last)` must be called with a keyword arg `last=<value>`
- default arguments must be at the end
- `*args` - enables you to pass any number of positional arguments
	- **args** is a **tuple**
- `**kwargs` - enables you to pass any number of keyword arguments
	- **kwargs** is a **dict**
- unpacking:
	- can unpack a list with  `foo(*my_list)` which will convert to positional args
	- can unpack a dict with `foo(**my_dict)` , keys must have the same names as parameters, length must also match
- local vs global variables
- call by object reference
	- mutable objects like lists or dicts can be changed within a method
	- immutable objects cannot be changed within a method
		- but immutable objects contained within a mutable object can be reassigned within a method
	- cannot rebind a reference within a method that was passed an argument

## The Asterisk* Operator

- can be used for:
	- multiplication and power operations, creation of lists or tuples with repeated elements, * args and ** kwargs, unpacking lists, merging containers into a list, 

## Shallow vs Deep Copying

## Context Managers

- tool for resource management
- allow you to allocate and release resources
- example: `with open('filename', 'w') as file:`
	- will close the file automatically and free up the resources
- examples include open/close db connections, locks for multithreading / multiprocessing
	- `with lock:`
- can create your own 
- 







#### 🧭  Idea Compass
- West  (similar) 
[[Python]] [[Python Syntax Basics]] [[Python Features Practical]] [[Python Programming Fundamentals Course]] 

- East (opposite)

- North (theme/question)

- South (what follows)