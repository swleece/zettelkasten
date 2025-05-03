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
[[StructureAndInterpretationOfComputerPrograms.pdf]]
[book](https://sarabander.github.io/sicp/html/index.xhtml#toc-Modularity_002c-Objects_002c-and-State)
[also book](https://sicpebook.wordpress.com/)
### 1A Overview and Introduction to Lisp

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


### 1B Procedures and Processes; Substitution Model 

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

### 2A Higher-order Procedures

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

### 2B - Compound Data

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
![[Pasted image 20241120190721.png]]

Remember the notion of **closure**, as a mathematician would say: "the set of data objects in Lisp is closed under the operation of forming pairs."
Many things ing computer languages are not closed, you may be able to make an array of ints but not an array of arrays.

A **list** in Lisp is a convention of representing a list as a sequence of pairs. E.g. a bunch of successive pairs where the car (first element) is the next value, and the cdr (second value) points to the next pair, terminated by a nil cdr.
There is a `list` operation to make this more ergonomic.

![[Pasted image 20241120190609.png]]

`map` higher order procedure, maps through a list recursively, applying an operation to each element
this is an example of capturing a general pattern of usage
"doing something to every element in a list"
could also be done in an iterative manner
key to the concept of **stream processing** 

`for-each` instead of acting on all elements on a list and returning the resultant list, instead take a list, act on the first element and return output, then continue to the next element of the list, etc.
very similar, but slightly different in what/how it returns

**metalinguistic abstraction**: 
a way to tackle complexity by building a suitable, powerful language with
- primitives
- means of combination
- means of abstraction

Henderson Escher Example
Ex: language create by Peter Henderson to display recursive images
there is only one primitive: a picture
- a picture simply draws a picture scalable to any rectangle
operations include:
- rotate, flip, beside, above
The closure property within these closed operations enables you to quickly grow complexity.

Language of Primitive Pictures > Language of Geometric Positions > Language of Schemes of Combination

contrast ^ this with the idea of defining all your specifications and sub tasks up front and attempting to build with a strict hierarchy

the design process not so much implementing programs but implementing languages to handle domain problems (and that's the power of lisp)

### 3B Symbolic Differentiation; Quotation

a robust system must be insensitive to small changes
solve a class of problems in the neighborhood of the main problem
create a language at the level of detail of the class of problems

representing derivatives and integrals
- rules of derivatives are reduction rules, perfect for recursion
- rules of integrals are not so simple but are complicated searches
(focusing on rules for derivatives in lecture for this reason)

Case analysis (corresponding to a large conditional)
- can write pseudo-code for all the rules of taking derivatives
- useful convention to use a `?` at the end of functions () 
- use `'` for accessing the symbolic object e.g. `'+` when checking if the operator of an expression is sum
	- e.g. `(EQ? (CAR EXP) '+)`


### 4A: Pattern Matching and Rule-based Substitution

Patterns -> Skeleton
- Rule: given a pattern, substitute into to get a new expression
- iow, pattern is matched against an expression source, result of application of the rule is a new expression (target) by instantiation of a skeleton
	- this describes the translation to get integral

Inventing a language:
- `?` - use to represent pattern variables, match to expressions
- `:` - stand for substitution objects, skeleton evaluations

- Pattern match:
	- foo - matches exactly itself
	- (f a b) - matches any list with first element is f, second is a, and third is b
	- (? x) - matches anything, call it x
	- (? c x) - matches a constant, call it x
	- (? v x) - match a variable, call it x
- Skeletons:
	- foo - instantiates to itself
	- (f a b) - instantiates to a 3 list, results of instantiating each of f, a, b 
	- (:x) - instantiates to the value of x in the pattern matched
```
(DEFINE DSIMP
	SIMPLIFIER DERIV-RULES)
```
- given a set of rules, produce a procedure that will simplify expressions containing the things related to those rules 


**Outlining the program:**

Control structure of pattern match substitution language
- several loops
- examine every sub-expression car cdr recursion, tree walk
- for every node, apply all the rules
	- every rule looks at every node
	- if the rule matches, replace the node with a new expression
		- simplify that new expression with the simplifier
		- then add that to the new expression
		- GIGO simplifier - build up from simple objects to simplify more complex objects
- recursive traversal 
- simplification


### 4B Generic Operators

Example:
Integrating HR systems for personnel across acquired companies.
Would like to be able to get Name, Salary, etc... generically.

Example:
Generic operators for doing complex number arithmetic. 
Imaginary number can be represented either as:
- a pair of real part and imaginary part
- a pair of magnitude and an angle

Rectangular vs polar complex numbers
we want operations that can add, subtract, multiply, divide across this abstraction barrier

real part, imaginary part, magnitude, and angle should be generic operators

Useful to have **typed data** , including both
- contents
- type

Use **type predicates** to see what the type of the data is

Conditionally use different methods for each type
Technique called **Dispatch on Type**
Frustrating that the types need to be managed conditionally, inflexible

Example:
**Generic Operators** and a Generic Arithmetic System
- use a table of operators and objects
- map given set of operator and objects to associated procedure
- operators and objects must still be typed
technique is called **Data Directed Programming**

*Paused at 45 minutes*

Example:
Generic Arithmetic System
Abstraction layer to enable add, subtract, etc. for any type of numbers (rational, complex, ordinary)
![[Pasted image 20250211211644.png]]
attach types to number object to enable correct routing across abstraction layer
chains of types enable you to go down into lower layers, types get built back up in the reverse direction

Example: Adding polynomials
use a typed data object, include variable, order, coefficient
reuses ADD from top level operations, applied to each order coefficients
Becomes a recursive tower of types as coefficients for polynomial can themselves be any number type
Also re-implement `*` as `MUL` such that you can handle polynomials in rational numbers
- this enables recursive combinations of all the number types 
Decentralized Control

Type coercion - the example as shown didn't exercise this, but it introduces a lot of real world complexity
Objects could have supported operations built into them in a more complex system
- e.g. greatest common denominator, only possible for some types of numbers
- each package would needs to define how to handle a new operator being added

## 5A: Assignment, State, and Side-effects

Only add a feature to a language if there is a good reason to

Assignment gives us another means of decomposition.
Thus far we've been writing functional programs, they encode mathematical truths
Processes evolved by such programs can be understood by substitution
Methods can be distinguished by the choice of truths expressed

`(SET! <var> <value>)`
! is just convention for an assignment
assignment introduces order dependence, <before> and <after> must now be considered

This now means the substitution method is no longer valid

Functional vs Imperative version of implementation of calculating 
- imperative approach introduces new potential ordering-related bugs

**Environment model of computation** (as opposed to substitution)
Let's provide terms for the names of things
- **Bound Variables**: like typical variable from calculus
	- lambda used as notation in computer science for bound variables
	- essentially, programs with different namings for bound variables are equivalent 
- **Free Variables**: 
	- for example, `*` or `+`, these name operators that are not interchangeable
- **Scope**: 
	- lambda expressions have scope

in lisp, you can compute with only lambda, all `defines` can get removed 

we now have names referring to places

**Environments**
environments are made of frames
a linked chain of frames
  ![[Pasted image 20250212195820.png]]

**Procedures**
![[Pasted image 20250212200036.png]]
The procedure is the whole thing, a compound object
- includes the code, something that can be executed
- includes the environment

The rules for evaluation for Environment Model:
Rule 1: 
A procedure object is applied to a set of arguments by constructing a frame, binding the formal parameters of the procedure to the actual arguments of the call, and then evaluating the body of the procedure in the context of the new environment constructed. 
The new frame has as its enclosing environment the environment part of the procedure object being applied.
Rule 2:

Assume there is a global environment include `+ - ( ) CAR` etc...
`DEFINE` is done relative to the global environment
enables you to add `MAKE-COUNTER` procedure to the global environment,
using that will create a new environment
and this can be instantiated multiple times
![[Pasted image 20250212201520.png]]
in this case, multiple, distinct instances of N are possible

each instantiation is an object is maintaining its own counter

Objects and Object Oriented programming
- assignment statement is useful for this and enables modularity

Example:
Using assignments to use implement cesaro's method for estimating Pi with monte-carlo experiments
- iterative procedure

## 5B: Computational Objects

we can inherit the modularity of the world into our programming via object oriented design

Ex: Electrical systems

## 5B: Computational Objects

example Digital circuits:
inverters
and-gates
or-gates
connected together with wires
these are all abstract variables
systems can be built with these primitives
we can use an embedded language in lisp to represent electrical diagrams digitally, procedurally

Implementing a Primitive: Inverter

inverter object needs to have an in wire that hooks back to the inverter and notifies the inverter of the change
the output wire needs to be able to tell downstream components that its value has changed
and-gates:
includes and and-gate delay
! used for assignment
and-gate action procedure

Simulation

Agendas (event-driven)
consists of header and queues at time intervals
![[Pasted image 20250222095801.png]]


## 6A: Streams, Part 1

**Stream processing**
**enumerator**, **filter**, **map**, **accumulator**
The above operators establish **conventional interfaces** (a new language) that allow you to glue things together.

Streams: a data abstraction used to make signal processing more modular, understandable, easy to work with
- head
- tail
- empty stream

Example: A stream where each element is itself a stream
would like to **flatten** the stream of streams
is just an accumulation of append-streams

flatmap - flatten of map 

8 Queens problem

Getting primes from a range of numbers

Stream programs can be written to run more efficiently by parallelizing the processing and piping output into the next function while the previous is still computing

Stream should contain the method of computing in it using procedures as first class objects

`DELAY Y` - takes an expression Y, produces a promise to compute the expression when you ask for it
- `(lambda() expression)`
- alternatively: `MEMO-PROC` (**memoization**) to cache the result, avoid computing the tail of the tail of the tail...
`FORCE X` - calls in a promise, computes the thing
- `(FORCE P) = (P)`

Delay decouples the apparent order of events from the actual order of events in the machine
Give up the idea that the visual of the program corresponds to the timing of execution

streams are a data structure built with an iteration control structure in it

## Part 6B: Streams, Part 2

Streams enable you to do on demand computation
only compute what you need when you need it

**Henderson Diagrams:**
- solid lines coming out are streams
- dotted lines are for initial values
![[Pasted image 20250323104850.png]]

**Sieve of Eratosthenes:** (method for finding primes)
- start with 2, cross out all subsequent integers divisible by two 
- then go to 3 cross out all subsequent numbers divisible by 3
- etc.

Sieves:
![[Pasted image 20250323104931.png]]

Stream programming:
- `add-stream` (like a finite state accumulator)
- `scale-stream`
- can program calculating fibonacci numbers

These are like recursive programming, recursively defined data objects instead of recursive procedures. 
There's no real difference between procedures and data.

**Applicative Order evaluation language**
- evaluate the arguments, then sub into body of procedure
vs
**Normal Order evaluation language**
- put promises (expressions) in to the body of the procedure until you get down to a primitive operator
- comes with the price of give up expressivity, hard to express iteration without growing state absurdly ('dragging tail problem')
- also does not mix well with side effects

The debate over **functional programming**
- does not have any side effects, more like mathematics than objects in the real world
- give up assignment for never having synchronization

Ex. see Cesaro test for random number in book
Ex. Banking system
- state of user, state of bank account
- instead of local state, bank account operates on stream of transactions, given an initial balance

Where fp breaks down?
- objects sharing state
- e.g. a joint bank account between two users 

How to define languages that can talk about delayed evaluation but also be able reflect the view that there are objects in the world?

## 7A: Metacircular Evaluator, Part 1

Until now, our notation has been a character/string description of a wiring diagram that could be written graphically.

**Universal Machine** 
`EVAL` - takes as input the description of a machine, such that it can emulate that machine
The evaluator for lisp - EVAL
```
(lambda (EXP ENV)
  (COND ((NUMBER? EXP) EXP) - special forms ...
		(( SYMBOL? EXP)(LOOKUP EXP ENVL))
		(( EQ? (CAR EXP) 'QUOTE) (CADR EXP))
		(( EQ?(CAR EXP) 'LAMBDA) 
			(LIST 'CLOSURE (CDR EXP) ENV))
		((EQ (CAR EXP) 'COND )
			(EV COND) (CDR EXP) ENV))
		((ELSE (APPLY (EVAL (CAR EXP ENV) - default
			EVLIST(CDR EXP) ENV)))))
```
`APPLY` - takes a procedure and applies it with arguments

Goes through the exercise of how to define the evaluator in lisp:
- define EVAL
- define APPLY
- define EVLIST
- define evaluation of conditionals
- define evaluation of binding variables
- lookup
- assq

This is the kernel of every language

Exercise: stepping through in detail an evaluation without assignment (side effects)

EVAL  produces a procedure and arguments (state) for APPLY
APPLY produces an expression and environment for EVAL
^ this is done in a loop essentially

## 7B: Metacircular Evaluator, Part 2

example: implementing a procedure that can take one required argument and n args
- goes over how to represent this syntactically using dot notation

Example
Dynamic Binding of Variables:
- problem of unbound variables when desiring to abstract 
- dynamic variable: a free variable in a procedure has its value defined in the chain of callers, rather than where the procedure is defined
- how do you guarantee that 
^Given up in exchange for the modularity principle
	- define PGEN
	- define term generation procedures
	- relies on returning procedures as values
















## Berkeley Lectures

[Berkeley Lectures](https://archive.org/details/ucberkeley_webcast_l28HAzKy0N8)

#### 🧭  Idea Compass
- West  (similar) 
[Computer Science]
[Programming]
- East (opposite)

- North (theme/question)

- South (what follows)