---
date: 2024-11-10
time: 20:14
note_maturity: 🌱
tags:
  - book
  - lecture
title: Structure and Interpretation of Computer Programs
author: 
link:
---
# Structure and Interpretation of Computer Programs

## MIT Course

[MIT Lectures](https://ocw.mit.edu/courses/6-001-structure-and-interpretation-of-computer-programs-spring-2005/video_galleries/video-lectures/)
by Hal Abelson and Gerald Jay Sussman
[book](https://sarabander.github.io/sicp/html/index.xhtml#toc-Modularity_002c-Objects_002c-and-State)
### Lecture 1A Overview and Introduction to Lisp

- Declarative Knowledge "what is true"
	- what you're looking for
- Imperative Knowledge "how to"
	- a method of how to do something

A process is directed by a pattern of rules called a procedure.

LISP - language

Techniques for controlling complexity:
- what computer science is actually about

**Black Box abstraction**
- enables modularization and suppression (abstraction) of details
- boxes can take in and output procedures (methods) 

**Conventional Interfaces** - agreed upon ways of plugging things together
- generic operations, large-scale structure and modularity
- oop
- operations on aggregates

**Metalinguistic Abstraction** - introspection into interpretation

When looking at or discussing a new language? What are the:
- **Primitive Objects**
- **Combinations** - are essentially trees, look like expressions in lisp
	- `(+ 3 12.4 5)`
	- consists of a combination an operator and operands
- means of **Abstractio n**
	- `(DEFINE A (* 5 5))` -> define A as 5*5
	- `(DEFINE (SQUARE X) (* X X))`
	- `(DEFINE SQUARE(LAMBDA(X) (* X X) ))`
- Conditionals
	- can use `cond` or `if`
```lisp
COND ((< x 0) (- x))
	((= x 0) 0))
	((> x 0) x))
	)
```
- Capture Common Patterns

**Block Structure** - packaging internal procedures inside a definition


### Lecture 1B Procedures and Processes; Substitution Model 

**Substitution Model** - model to understand how procedures and expressions yield processes 
- example: basically substitute evaluated expression until you reach the result

Kinds of **Expressions**:
numbers, symbols, lambda-expressions, definitions, conditionals, combinations

Intuition: 
- going from shapes of programs to get shapes of processes
- visualize the depth of expressions and the count of expressions evaluated to get a result
	- corresponds roughly to time and space O(x), O(1)
- **Iteration** vs **Recursion**
- consider how in-memory state
- **perturbational analysis**

Fibonacci Numbers
- recursive calculation of fibonacci numbers

Towers of Hanoi

### Lecture 2A Higher-order Procedures

**Higher-order Procedures**: procedures that take procedural arguments and makes procedural values

Some abstractions, some which cannot be done easily in languages other than Lisp

Abstractions are useful for storing procedures for solving certain problems that are hard to reason out
Look for patterns that should be abstracted, watch out for when you are repeating very similar code

define SUM-INT, define IDENTITY,
define PI-SUM

**Abstractions** serve to make programs more easy to write and more easy to read.

Heron's method for the square root
- looking to find a fixed point
- define sqrt x, is the fixed-point of that procedure which takes an arg y and averages x/y with y, starting with 1

```Lisp
(DEFINE (FIXED-POINT f start)
		(DEFINE (ITER OLD NEW)
			(IF (CLOSE-ENUF? OLD NEW) NEW (ITER NEW (F NEW)))))
```

Wishful thinking when writing programs, write pseudo-code and fill in the implementation of lower level logic later.

Newton's method for computing the derivative of a function

"The rights and privileges of first-class citizens"
- To be named by variables.
- To be passed as arguments to procedures.
- To be returned as values of procedures.
- To be incorporated into data structures.
	- Chris Strachey

^ w.r.t. procedures as first-class citizens

### Lecture 2B - Compound Data

Abstractions enable a layered system that enable levels of hidden detail.
Divorce the task of building things from the task of implementing the parts.
The same concept can apply to data. Building up data in terms of simpler data.

Primitive Data
Combinations of primitive data -> compound data

Ex: Arithmetic on rational numbers
- 3/4 * 2/3 = 1/2 (use rules we learn for computing fraction operations, can be rewritten as operations on numerators and denominators)
- Again use Wishful Thinking (imagine process, write as pseudo-code, work down to lower levels of abstraction until implementation)
- Ideally the programming language should reflect the concepts we hold in our heads.
- when building a solution, it's valuable to encapsulate both numerators and denominators in one object instead of separating them and having variables for each inputs numerators and denominators
- encapsulating/packaging related data together is essential for proper abstraction
- there is an abstraction layer between the use of data objects and the representation of data objects
	- with translating between `pairs` and rational number operations
	- known as **Data Abstraction**: methodology of setting up objects to separate use from representation, through the use of compound data objects
- isolation enables alternative representation

System designers should generally aim to maintain flexibility untIl they need to make decisions.
This is helped by data abstraction.

**List Structure**: provides a way to package pairs in Lisp
```List
pair - a primitive data type, 
cons - constructs a pair
car - selects first part of pair
cdr - selects second part of pair
```

**Box and Pointer Notation**: refers to the notation of using boxes to represent values or data stored in memory, arrows represent references to another memory location. Commonly used to represent lists, trees, graphs, arrays

`let` vs `define` : let enables you to set up a local context with a definition

The value of data abstraction is shown by how well you can use code as building blocks for solving other problems.

Ex. representing vectors in a plane

**Closures** - the means of combination in your system are such that when you put thiNgs together, you can then put those things together with the same method, (i.e. you can have a pair of numbers, or you can have a pair of pairs)

**Abstract Data**: 
- a contract is satisfied by the data abstraction
- if x = (make-rational-num N D)
	- THEN (numer x) / (denom x) = N / D

What are `pairs` really?
Axiom for pairs:
- for any x and y
- car of cons of x, y is x
- cdr of cons of x, y is y
(actually pairs can be built out of anything or nothing at all)

- rational numbers built on top of pairs 

We're going to blur the line between what's considered data and what's considered procedure

### 3A Henderson Escher Example

Methodology of data abstraction from last lecture
- isolate data objects usage from representation
- there are ways to glue together data objects 

Example, representation of vectors using pairs, definitions of vector operations

Get used to procedures being name-able objects

(2, 3) -> (5, 1)
can represent as a pair of a pair of pairs

Remember the notion of **closure**, as a mathematician would say: "the set of data objects in Lisp is closed under the operation of forming pairs."
Many things ing computer languages are not closed, you may be able to make an array of ints but not an array of arrays.

A **list** in Lisp is a convention of representing a list as a sequence of pairs. E.g. a bunch of successive pairs where the car (first element) is the next value, and the cdr (second value) points to the next pair, terminated by a nil cdr.
There is a `list` operation to make this more ergonomic.

`map` higher order procedure, maps through a list recursively, applying an operation to each element
this is an example of capturing a general pattern of usage
"doing something to every element in a list"
could also be done in an iterative manner
key to the concept of **stream processing** 

`for-each` instead of acting on all elements on a list and returning the resultant list, instead take a list, act on the first element and return output, then continue to the next element of the list, etc.
very similar, but slightly different in what/how it returns




























## Berkeley Lectures

[Berkeley Lectures](https://archive.org/details/ucberkeley_webcast_l28HAzKy0N8)

















#### 🧭  Idea Compass
- West  (similar) 
[Computer Science]
[Programming]
- East (opposite)

- North (theme/question)

- South (what follows)