---
date: 2024-07-27
time: 11:13
note_maturity: 🌱
tags:
  - project
---

# Python Learning via Chat

# Intermediate Python Concepts and Techniques

1. Object-Oriented Programming (OOP) in Python
   1.1. Classes and Objects
   1.2. Inheritance and Polymorphism
   1.3. Encapsulation and Data Hiding
   1.4. Magic Methods (Dunder Methods)

2. Advanced Function Concepts
   2.1. Decorators
   2.2. Closures
   2.3. Lambda Functions
   2.4. Function Annotations

3. Iterators and Generators
   3.1. Creating Custom Iterators
   3.2. Generator Functions and Expressions
   3.3. The `yield` Keyword
   3.4. Coroutines

4. Context Managers
   4.1. The `with` Statement
   4.2. Creating Custom Context Managers
   4.3. The `contextlib` Module

5. Advanced Data Structures
   5.1. Collections Module (deque, Counter, OrderedDict)
   5.2. Namedtuples
   5.3. Heapq Module for Priority Queues

6. File I/O and Serialization
   6.1. Working with CSV and JSON
   6.2. Pickling and Unpickling Objects
   6.3. Working with Binary Files

7. Functional Programming
   7.1. Map, Filter, and Reduce Functions
   7.2. List Comprehensions and Generator Expressions
   7.3. Partial Functions

8. Error Handling and Debugging
   8.1. Custom Exceptions
   8.2. Context Managers for Error Handling
   8.3. Debugging Techniques and Tools

9. Concurrency and Parallelism
   9.1. Threading and the Global Interpreter Lock (GIL)
   9.2. Multiprocessing
   9.3. Asyncio for Asynchronous Programming

10. Module and Package Management
    10.1. Creating and Importing Modules
    10.2. Package Structure and `__init__.py`
    10.3. Virtual Environments and Dependencies

[**Descriptors**](https://docs.python.org/3/howto/descriptor.html): 
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

Python Attribute lookup process: When you access an attribute using the dot notation (like `a.x`), Python follows a specific lookup order: a. It first checks the instance dictionary. b. If not found, it then looks in the class dictionary. c. If still not found, it checks any base classes (following the method resolution order).

**Abstract Base Class (ABC)**:
Abstract base classes complement [duck-typing](https://docs.python.org/3/glossary.html#term-duck-typing) by providing a way to define interfaces when other techniques like [`hasattr()`](https://docs.python.org/3/library/functions.html#hasattr "hasattr") would be clumsy or subtly wrong (for example with [magic methods](https://docs.python.org/3/reference/datamodel.html#special-lookup)).
ABCs introduce virtual subclasses, which are classes that don’t inherit from a class but are still recognized by [`isinstance()`](https://docs.python.org/3/library/functions.html#isinstance "isinstance") and [`issubclass()`](https://docs.python.org/3/library/functions.html#issubclass "issubclass"); see the [`abc`](https://docs.python.org/3/library/abc.html#module-abc "abc: Abstract base classes according to :pep:`3119`.") module documentation. 
Python comes with many built-in ABCs for data structures (in the [`collections.abc`](https://docs.python.org/3/library/collections.abc.html#module-collections.abc "collections.abc: Abstract base classes for containers") module), numbers (in the [`numbers`](https://docs.python.org/3/library/numbers.html#module-numbers "numbers: Numeric abstract base classes (Complex, Real, Integral, etc.).") module), streams (in the [`io`](https://docs.python.org/3/library/io.html#module-io "io: Core tools for working with streams.") module), import finders and loaders (in the [`importlib.abc`](https://docs.python.org/3/library/importlib.html#module-importlib.abc "importlib.abc: Abstract base classes related to import") module). You can create your own ABCs with the [`abc`](https://docs.python.org/3/library/abc.html#module-abc "abc: Abstract base classes according to :pep:`3119`.") module.