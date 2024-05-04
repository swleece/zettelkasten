---
date:
  "{ date }": 
time:
  "{ time }": 
note_maturity: 🌱
tags:
  - productivity
---
# Neovim Commands Learning Aid

**Text Objects**: specific, predefined parts of the text that can be easily selected, manipulated, or operated upon with various commands
- they let you execute commands on semantic units within your file
- **Pair Text Objects**: characters that usually come in pairs, like `()`, `{}`, `[]`, and `""`
- **word** (`w`): There are also text objects for words
- **WORD** (`W`): include all non-blank text
- **sentence** (`s`): sentences are defined by punctuation marks (. ! ?) followed by whitespace or a newline
- **paragraph** (`p`): a block of text separated by one or more empty lines containing no characters or only whitespace characters
- **Custom Text Objects**: users can define custom text objects based on specific criteria, such as function blocks, HTML tags, and more.

## Motions

`h j k l`
`0` - jump you to start of line
`w` - until the start of the next word, *excluding* its first character
`e` - to the end of the current word, *including* the last character
`$` - to the end of the line, *including* the last character 

`%` - use to find / jump to a matching `)`, `]`, or `}`

`{`, `}` - as a motion, jump backward / forward *paragraph*
`(`, `)` - as a motion, move cursor to beginning of previous / next *sentence*

## Operators

- typically can be followed by a number to execute that number of times (e.g. `d2w` to delete two words)
`d` - delete

`c<motion>` - change operator, deletes *to* the provided motion and changes mode to insert (e.g. `ce` to delete rest of word and enter insert mode)

`y` - yank operator, copy text to enable you to `p` put it

`<operator>i` - inside, e.g. `di)` to delete inside current parentheses
`<operator>a` - around, e.g. `da)` to delete around (inclusive) parentheses

## Commands

`:` - 
Command completion:
- while typing a command after `:`, use `<C-d>` to show possible completions
`<C-w>` - window management generally
- `<C-w><C-w>` - jump between windows
- `<C-w>(hjkl)` - jump between windows


### Text editing and motion commands

> Many commands that change text are made from an **operator** and a **motion**.

`x` - delete character under the cursor
`dw` - **d**elete a **w**ord
`dd` - delete current line (`2dd` to delete two lines)
`d$` - delete to the end of the line
`u` - undo the last command
`U` - undo all the recent changes on the current line

`p` - put command, place previously deleted (or yanked) text after the cursor
`P` - put previously deleted text before the cursor
`r` - replace, replace character at current cursor position with provided character
- related:`R` - enter replace mode to replace more than one character until you escape

`o`/`O` - open a line below/above the cursor and enter insert mode

`<C-g>` - show your location in a fIle and the file status
`G` - jump to bottom of file
`gg` - jump to top of file

`:s` - substitute command
- `:s/old/new/g` to substitute "new" for "old"
	- the `g` flag means to substitute globally *in the line*
- `:<#>:<#>s/old/new/g` to do a substitution across a range of line numbers (where the `#`'s are two line numbers, inclusive)
- `:%s/old/new/g` to substitute every occurrence in the whole file
- `:%s/old/new/gc` to find every occurrence and ask for **c**onfirmation whether to replace or not

### Search
`/` - search command, 
- `n`, `N` - go the next / previous occurrence of the searched string
- `?` - search command as well but searches in the opposite direction
- `<C-o>` - return to where the cursor started before you entered search
- `<C-i>` - move backward(?)

`%` - use to find a matching `)`, `]`, or `}`

`:s/(pattern)/replacement/` - using parentheses, the pattern can be used in the replacement (note: find a real example)

## Modes

### insert

`i` - enter insert mode at current cursor location
`a` - move cursor forward one character and enter insert mode
`A` - move cursor to end of line and enter insert mode 

### normal

`<Esc>` - to return to normal mode

### visual

`v` - enter visual mode

### replace

`R` - enter replace mode


## Options

`:set <option>`

Options:
- `ic`, `noic` - ignore case when searching (`ignorecase`, `noignorecase`)

To invert the value of a setting, prepend it with `inv` (e.g. `invignorecase`)


## External Commands

`:!` + external command to execute an external command
- e.g. `:!ls` to show listing in current directory













#### 🧭  Idea Compass
- West  (similar) 

- East (opposite)

- North (theme/question)
[[productivity]]
- South (what follows)
