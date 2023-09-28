---
date: 2023-09-19
time: 15:14
note_maturity: 🌱
tags:
---
# Python Syntax Basics

Function parameters can have default values, if function is called with no parameters given, the default values will be used.
`def f(x= 0, y = 0)`

Python uses (only) indentation to delimit blocks of code. 

Print converts numbers to strings.

Format strings: string with curly braces for replacement. '.format' specifies data for replacement
`'my name is {}'.format('Stephen')` my name is Stephen
`'I have {0:2d} cats named {2} and {1}'.format(2, 'name1', 'name2')` I have 2 cats named name1 and name2
`'I have ${:.2f} in my pocket'.format(5.125)` I have $5.12 in my pocket


Example code visualization
![[Pasted image 20230919151601.png]]


## Conditional Expressions

evaluate to true or false

gt >
lt <
equal to ==
not equal to !=

else statement optional

elif (else if) statements optional

pass means do nothing, just helps readability
By convention, pass is used to indicate the end of a block. (unless a return statement would otherwise go there)
## Logical operators

x and y
x or y
not x

## Loops

`while
- when you reach the end of the loop body, return execution to the beginning, until condition not satisfied
```
while a < b
	a + 1
	pass
```

`for
Range function often used in for loops
generates a range object
`range([start], stop, [step])`
- optional start and step parameters

`for i in range(1, n)

### Strings 

A sequence of characters.
In python, strings are a particular type of object.

The dot operator can be called on objects to access methods and data.

Brackets `[]` can also be used to access values at valid indexes.

### Lists and Tuples

Python has two ways to construct sequences of data.
Both allow you to write down multiple data items in one object that contains those elements.

Tuples typically used to group together data that form a logical whole
Tuples can hold different types of data.

Syntax:
Tuples are surrounded by `()`
Empty tuple can be created. 1 element tuple must have comma
`tuple('abc')` is like `tuple('a', 'b', 'c')`
Tuple data stored in the heap and has a pointer from variable name to it.

Semantics:
can access elements using square brackets `[]`

Tuples can be iterated over `for i in exampleTuple`

Tuples can be unpacked to multiple variables:
`c = (1, 'two', 3.2)
`x, y, z = (c)
-> `x = 1` and `y = 'two'` and `z = 3.2`

Elements can be accessed using brackets `[]`

Lists are *mutable* while tuples are *immutable*

Pointer to tuple returned by function: (last line is `ans = f(c)`)
![[Pasted image 20230919215851.png]]

### More on Lists

Lists are sequences of multiple items
Can add more items to it after it has been created.

### References to lists

Lists are created in the **heap**
- outside of any frame (persists after a function return)
- the box contains a pointer to it
![[Pasted image 20230921211854.png]]

The value of variables assigned to lists is actually the pointer (reference) that points to the list in the heap.

Methods can modify an object 
- somelist.append(item) (or insert or remove)
- somelist.sort()

other methods can return a value, leaving the object unchanged
- somelist.copy()
- sorted(somelist)

### Iterating over lists

`for x in someList:`
	`print(x)`

- note that x remains in scope and continues to reference last item of the sequence

A list can contain references to other lists 
![[Pasted image 20230921212634.png]]
-  in this example, we iterate over those references
 
### List Indexing

Indices start at 0
`myList[0]`

### List Slicing

inclusive at start, exclusive at end
- start/end positions are optional (behave as expected)
- if both left blank, entire list is copied

`lst1 = [1, 2, 3, 4, 5]`
`lst2 = lst1[1:3]` -> `[2, 3]`

^This copies elements in that range to a new list
- i.e. `lst2` points to a new, two element list in the heap `[2, 3]`

Can also replace elements using slice range
`lst1[1:3] = [0]` -> `[1, 0, 4, 5]`


## Default Arguments

As a general rule, only use default arguments that are immutable
- i.e. don't use lists (also sets, maps, objects of your own class)
- Can use strings, tuples, and numbers














#### 🧭  Idea Compass
- West  (similar) 
[[Computer Science]]
[[Python Programming Fundamentals Course]]

- East (opposite)

- North (theme/question)

- South (what follows)
