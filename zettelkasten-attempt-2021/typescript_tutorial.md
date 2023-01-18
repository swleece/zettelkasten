## Notes on a youtube tutorial for Typescript

javascript is dynamically typed, 
typescript is a tool that applies static typing rules to javascript in development,
  then throws errors when compiled to javascript.

typescript has many more types that javascript and enables the creation of your own types. 

The core job of typescript is to check types and alers us if something is wrong.

Core Types:
  Shared with javascript:
  number (same as javascript, all numbers are floats by default) (e.g. 1, 5.3, -10)
  string ('Hi', "HI", `Hi` <- dynamic literal)
  boolean (just these two, "truthy" or "falsy" values come later at a higher level)
  
  object (any javascript object is of type object) 
  array ([1, 2],  (can have array of mixed data)), types of arrays can be flexible of strict

typeof operator : returns type of object

Typescript types:
  E.G.:
    const person {
      name: string;
      age: number;
    }
  include key-type pairs (lots of colons)
  best practice is to allow typescript to infer type where possible



