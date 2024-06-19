---
date: 2024-01-28
time: 10:26
note_maturity: 🌱
tags:
  - youtube
  - flashcards
link: https://www.youtube.com/watch?v=LfaMVlDaQ24
---
# Harvard CS50 (2023)

## Programming in Scratch

## Programming in C

### General

`clang -o hello hello.c -lcs50`
`make hello.c` -> `hello`
both of these compile hello.c, which includes link to external package cs50

### four steps to compiling

1. preprocessing
	- preprocessor directives 
		- including `#` -> `/usr/include`
		- operates like a global find and replace, providing files included
2. compiling
	- after preprocessing, only C code remains
	- this step converts the C code to Assembly language code
		- `main`, instructions, moving values around, arithmetic to move things in and out of memory
3. assembling
	- conversion to actual zeros and ones
4. linking
	- links all the included code and the original file's code
	- combines new, current file code, library code, and C library code

decompiling also possible, not as threatening as you might think
- challenges to reverse engineering
	- challenges with intellectual property
	- variable names not retained, function names not retained
	- for loops and while loops can look exactly the same
	- it's about as hard as just building it yourself

### debugging

- can use `printf`
- often better to use a debugger
	- debuggers are special programs for running your code
	- play through to the end
	- step over will run one line at a time
	- step into will step into the execution of a function line by line
	- given access to variables, call stack, flow of your program

Rubber ducking
- talking out your problem / solution to figure out the problem
- i.e. sounding out your problem

### data types, arrays, and memory management

bool, int, long, float, double, char, string, etc.
each has a specific number of bits / bytes used

[[typecasting]] will convert something to a new type if possible

arrays are more memory efficient than a bunch of variables
- `int scores[3]` - declaring array of ints with length three
- the memory for an array is contiguous

magic number: something that should be declared at the top of the file so that you only need to modify it in one place in the future
- for example, the length of an array of test scores

strings are variable length
- strings are actually arrays of char's 
- an extra byte at the end of the array, sentinel value, as a delimiter for the end of the array, `\0`

technically, `\0` maps to `NUL`

`.h` - header file, useful libraries
- string.h, ctype.h, stdio.h, cs50.h

command line arguments can be accessed 
```C
int main(int argc, string argv[])
{
	printf(argv[1])
}
```

exit status:
- whenever program finishes, returns an integer
	- notice the int in int main
- `echo $?` - bash command to get the exit code
- use return values in program to give return value from program

cryptography:
- plain text + key -> cipher text using an algorithm
- cipher
- e.g. shift each letter by 1 (a -> b), key = 1, cipher is add 1

## Algorithms

common algorithmic complexity:
n^2, n log n, n, log n, 1

## Search

linear search:
- loop over entire array
- O(n)
binary search:
- requires array to be sorted
- O(log2n)  -> O(log n)

O (worst case) vs Omega (best case)(often 1)

Theta: represents upper bound and lower bound are both the same
- for example, counting everyone in a room 1 by 1

in C, you can create your own data structure with a [[struct]]
```C
typedef struct
{
	string name;
	string number;
}
person;
```
e.g.
```C
person people[2];
people[0].name = "Carter";
people[0].number = "+1-617-495-0000"
```

[[Selection sort]]
- O(n^2)
- Omega(n^2)

[[Bubble sort]]
- O(n^2)
- Omega(n)

[[Merge sort]]
- O(n log n)
- Theta(n log n)

in order to improve time (execution), need more space (memory)



## Memory

0123456789ABCDEF
hexadecimal, base 16
hexadecimal digits can be represented with 4 bits
11111111 maps to FF : 255
locations in memory designated with hexadecimal since it is more compatible with binary than base 10
to denote hexadecimal, prefix hexadecimal numbe with `0x`

`&` - reference, enables you to get the address of a variable in memory

```C
int n = 50;
printf("%p\n", &n); # print the reference to n
# could yield 0x7ffcc784a04c
```

`*` dereference operator, enables you to go to an address in memory

pointers : a variable that contains the address of some value
- or can be thought of as the address of something in memory
```C
int n = 50;
int *p = &n;
```

pointers typically take 8 bytes, integers typically require 4 bytes
- can be thought of and used as a variable that points to another address

double pointers `**` do exist

Strings: `""`
- `string s = "HI!"`
- s is actually a pointer to the where the string is actually stored in memory
- in other words, the above actually corresponds to `char *s = "HI!"`
- string is not actually a data type in C, `char *` is actually what was made using `typedef`

`*` is used in 2 ways, different when you declare a pointer use datatype and \*, vs when you use a pointer you just use \*

arrays are stored consecutively in memory
strings are arrays of chars that end with `NUL`

`malloc()` - explicitly ask the OS the allocate the amount of memory requested
`free()` - free the memory previously requested
`NULL` - by convention, the first memory location. not ever used by convention, instead used when memory could not be allocated

`valgrind` - tool for debugging, identifying leaks

garbage values - values in memory that was previously allocated 

memory used by machine code, globals, heap and stack
![[Pasted image 20240226204536.png]]
- heap memory allocated downward as pointers and values are set
- stack memory allocated up as functions are called

Buffer overflows:
- Stack overflow
- Heap overflow

## Data Structures

**abstract data types**: enable defining custom data structures

**queue**: FIFO, first in first out
- enqueue - add to the *end* of the queue
- dequeue - remove from *front* of the queue
flashcard: data structure, first in first out;; queue
<!--SR:!2024-06-29,10,270-->

**stack**: LIFO, last in first out
- push - add to the *top* of the stack
- pop - remove from the *top* of the stach
flashcard: data structure, last in first out;; stack
<!--SR:!2024-06-28,9,250-->

```C
typedef struct
{
	person people[CAPACITY];
	int size
}
```

`malloc` - memory allocate
`realloc` - handles the copying of an existing memory allocation

`struct` syntax: 
`->` : go to the location a pointer points to and get it's value

### linked list:  
- requires linking list elements with a pointer to the next value
- ends with null, `0x0`, aka `NULL` 
- finally, just need a pointer to the start of the chain to be the reference to the linked list
![[Pasted image 20240307212743.png]]
or
![[Pasted image 20240307212821.png]]
```C
typedef struct node
{
	int number:
	struct node *next;
}
node;
```

Running times for linked list operations:
- searching for a value: O(n)
- insertion via prepend (stack): O(1)
- insertion via append (stack): O(n) (but could be easily O(1)
- insertion with sorted order maintained: O(n)
	- i.e. insert 3 between 2 and 4 in `[1,2,4]`

### Binary Search Trees

introduces another dimension to the data structure where at each node two pointers can be followed
```C
typedef struct node
{
	int number:
	struct node *left;
	struct node *right;
}
node;
```

```C
bool search(node *tree, int number)
{
	if (tree == NULL)
	{
		return false
	}
	else if (number < tree->number)
	{
		return search(tree->left, number);
	}
	else if (number > tree->number)
	{
		return search(tree->right, number)
	}
	else
	{
		return true;
	}
}
```


### Dictionaries

- key value pairs

**hashing** : aka *bucketizing*, analogized as grouping a bunch of randomly ordered playing cards by suit before sorting by order 
**hash function** : converting something bigger to something smaller
**hash tables** : an application of arrays and linked lists to achieve this

Example: contacts list
- composed of an array of letters where each letter points to the start of a linked list of contacts starting with that letter
- additionally, can keep including more letters in key to get closer to constant time
![[Pasted image 20240317094342.png]]

where the **hash table** is the array of letter(s) pointing to linked lists

potentially O(n), requires uniform distribution ideally

### Tries

short for retrieval
a tree where each node is an array

![[Pasted image 20240317095157.png]]

enables us to get constant time at the expense of memory


## Python

- where C is a compiled language, python uses an interpreter to interpret your code
- python is dynamically typed, will infer data types
- don't have to manage your own memory
- don't need to use a `main` function necessarily
	- however there is a convention to define a function called main placed at the top and call it at the bottom `main()`
- integers automatically converted to a float
- python does not have pointers that you manage manually
### types

- bool
- float
- int
- str
- range
- list - 
- tuple - 
- dict - 
- set - collection of values, gets rid of duplicates

**truncation**: done automatically for floats
**floating-point imprecision**: still an issue out of the box
**integer overflow**s: addressed in python, not a problem. still limited in how many digits will be shown on screen though

strings in python are **immutable**:
- `s = s.lower()` returns a copy of the string not the original, the original bytes get freed up automatically later
### object-oriented programming

- focused on procedural programming in C, using functions to update values
- would be nice if some data types had functionality built-in
- certain data types have methods (functionality) built-in
	- for example, the data type `str` has functionality that is built-in like `example_string.lower()`

### Scope

variables are not scoped

`try` `except` `else`: if something goes wrong in `try` block, will return `except`
- best effort to get what you're trying to do, is pythonic
- `else` is also available to `try` block, technically different than the usage under `if`
- `try` *should* wrap only the fragile, but may be nicer to wrap more logically

default values and named arguments

strings:
- can use multiplication for multiple concatenations

### More handy functions for specific data types

- list methods
```Python
a_list = []
sum(a_list)
len(a_list)
a_list.append(a_number) # OR a_list += [a_number]
```
- lists can be accessed by slice
- lists can be easily linear search with `if name in a_list:`

- strings can be iterated over

- dicts: `a_dict = {"Carter": "1234", "David": "5678"}`
	- can get constant time access via `if name in a_dict:` 
	
### sys

library that comes with python and can be used to access the system
- `argv` - can use to access command line arguments
- `sys.exit(1)` - enables you to exit the program, providing an exit code

**swapping**: `x, y = y, x` will swap two values

### Working with files

`import csv` - there are many useful libraries for working with files

`with` keyword
- will automatically close a file after it is opened within a block
```Python
with open(...):
	...
```

## SQL





























#### 🧭  Idea Compass
- West  (similar) 
[[Computer Science]]
- East (opposite)

- North (theme/question)

- South (what follows)