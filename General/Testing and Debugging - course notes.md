---
date: 2023-09-21
time: 12:11
note_maturity: 🌱
tags:
---
# Testing

[[Python Programming Fundamentals Course]]

Testing: finding bugs
Debugging: fixing bugs

Test cases should test hard cases, not just easy ones.

## Black Box Testing

Only consider the expected behavior of the function--not any implementation--to devise test cases.
It's important to test corner cases.

Consider error cases, what inputs are invalid and why?
Can you make them as close to valid as possible?
Only test 1 error at a time

Code should handle error cases

Test valid inputs as well.
Testing hard, valid cases important as well. 

## White Box Testing

Examine code, focus on code coverage.
Parts of code not run by tests -> more likely to have problems.

## Test Coverage

### Statement coverage

Full coverage means every statement in the function is executed.

### Decision coverage

Full coverage means all possible outcomes of all decisions are exercised.
Consider visualizing with a [[control flow graph]] 
- a directed graph (mathematical) whose nodes are basic blocks and whose edges represent the possible ways the control flow arrow can move (arrows)
- decision coverage corresponds to having a suite of test cases which covers every edge in the graph
### Path coverage

Full coverage means test cases must span all possible valid paths through the [[control flow graph]].
This grows exponentially.

## Creating Test Cases

- make sure tests cover every error case
	- valid inputs
- test 'too many' as well as 'too few' (n + 1, n - 1)
- any given test can only test one "error message and exit" condition
- test at the boundary of validity
- consider any valid input that have special treatment
	- e.g. Aces in poker can be either high or low
- consider types
	- integer instead of float?
	- 32-bit int instead of 64-bit
- off by one errors
- consider all major categories of inputs
	- numerical: negative zero positive
	- sequences of data: empty sequence, single element, many element
	- characters: lowercase, uppercase, digits, punctuation, spaces, non-printable
	- categories relevant to the algorithm: e.g. whether a number is prime
	- combinations of the above
- if you can list all possible answers, your tests should include all possible answers

## Generating Test Cases

In some cases, you can test all inputs over a range to ensure you don't miss any important ones.
In other cases where there are too many possible inputs, you can pseudo-randomly test as many cases as possible.
- [[Law of Large Numbers]] makes it likely that you will encounter a lot of varieties of relationships between parameters

In order to do this, you may need to implement a simple but slow (brute force) implementation to test against a more complex one (in order to find the answer to test against).
Sometimes the properties of the answer can be used to verify a correct answer
- e.g. square root of n -> n squared

An [[test harness]] is a program used to run and test the main parts of your code.

## Asserts

`assert expr` - checks that the expression evaluates as true
- if `expr` is false, an error is raised
It's almost always better to detect an error and crash than to give a wrong answer.

the `-O` option disables debugging features such as assertions and can be used for performance improvement

## Code Reviews

Reviewers have fresh eyes and may see problems/cases you didn't see.
Explain the code line-by-line.


# Debugging

How not to debug:
- change something and hope it works

Use the [[Scientific Method]]:
- Observe a phenomenon
- ask a question
- gather information, apply expert knowledge
	- iterative information gathering
- form a hypothesis (clear statement of what is wrong in the case of programming)
- test hypothesis (try a particular input, examine internal state during execution)
- Refute (and repeat above) or accept hypothesis (and move forward)

## Hypotheses:

Ex: my program crashes if the input is too long
- better: ... if the input is more than 200 chars
- better: ... because on line 57 I have a fixed sized, 200 element buffer

Hypothesis should be:
- Clear - precise
- Testable - refutable
- Actionable - if accepted, we can do something useful

## Confounding variables

Watch out for possible alternate explanations or confounding variables
- characters - could be character types or count

Print statements are useful

Many programs are deterministic (single-threaded), which makes debugging easier
- not so for multi-threaded


# Types of Bugs

### Syntax errors

Errors that prevent your code from running
- indentation errors
- not including correct punctuation

Often these will be surfaced in your editor by linters before you even run your code.

### Logic errors

Valid code that simply does not do what it was intended to do.


# Debugging Tools

depends on taste and the type of code

### Using print() statements

Always available

Python has a useful builtin shortcut for printing any variable:
e.g. `f"{x=}` -> prints `x=42`

### Debuggers

Very powerful but require setup

enable you to step through execution and access all variables and function calls as they evolve

Tip:
available in the terminal with vanilla Python using a package `pdb`

### Debugging in VS Code

- have python vs code extension
- make sure terminal is visible
- check python interpreter and environment
	- make sure the environment has the tools you need
- test some simple code (using play button)

Use Run and Debug (debugging environment):
- set a breakpoint or multiple (code will pause when reached) by clicking in the gutter
	- default breakpoint is for "Uncaught Exceptions"

### for .py files

1. Debug controls
	1. continue
	2. step over : and stop at the next line of code
	3. step into : and show the execution within that function rather than in the parent function
	4. step out : of a child function back up to the parent function
	5. restart : execution
	6. stop : execution
2. Execution line indicator : tells the next line that will be executed
3. Variable explorer panel : shows a list of panels either in local scope (current function). Also variables in global scope. variables are editable in memory
4. Watch Panel : enables you to create custom expressions that are evaluated throughout the execution of the program to monitor them throughout
	1. e.g. enables you to filter 1,000 variables to only a few
5. Call stack : clicking around here enables you to access different points in the call stack
6. Debug console : enables you to interact with variables at any step in the execution for which execution is paused. Allows full interactive access to the current state of the variables in the midst of execution
	1. can introduce new variables entirely
	2. can edit variables

Data viewer: (right click variable -> View in Data Viewer) 
- enables you to view data in a grid form 

Hover in editor:
- when a breakpoint is reached, you can hover over a variable in the editor and see the current value as a tooltip

### For .ipynb files (jupyter notebooks)

VS Code provides many of the same features for Jupyter notebooks

breakpoints, debug session, and run-by-line accessible in any code cell

Jupyter: variables console
- similar to the debugging variable explorer, but available outside of a debugging session
- also have access to the data viewer tool (for exploring matrices)













#### 🧭  Idea Compass
- West  (similar) 
[[Python Programming Fundamentals Course]]
[[Computer Science]]
[[Software testing]]
- East (opposite)

- North (theme/question)

- South (what follows)
