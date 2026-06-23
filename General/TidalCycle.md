---
date: 2025-11-16
time: 16:25
note_maturity: 🌱
tags:
  - project
---
	
# TidalCycle

https://tidalcycles.org/docs/patternlib/tutorials/workshop#gain-pitch-and-panorama

See also [[Strudel]] - https://strudel.cc/
- browser based

### Launch SuperCollider in terminal (sclang)

`cd /Applications/SuperCollider.app/Contents/MacOS`
`./sclang "$@"`

SuperCollider application/IDE can also be launched, note that it seems the SuperCollider IDE does not work with the neovim configuration. But does work with the (recommended) Pulsar text editor.

### NeoVim plugins

Launcher: tidal.nvim - https://github.com/thgrund/tidal.nvim
- sends chunks of TidalCycles code to Tidal and SuperCollider interpreters
	- use `leader CR` to send line
- should do highlighting, but haven't gotten that working

Text editing macros: tidal-makros.nvim - https://github.com/thgrund/tidal-makros.nvim

### Other Notes

Need to open Audio Midi setup app and adjust settings of output rate for my over-ear headphones.


