---
date: 2022-12-06
time: 08:33
tags: youtube, zettelkasten, notes

---

[video](https://www.youtube.com/watch?v=wB89lJs5A3s)

Title : My 2021 Comprehensive Obsidian Zettelkasten Workflow


### Takeaways:
- don't try to make a highly structured knowledge tree (i.e. directory structure 
- consider prepending input types with designated character
	- "{ " for books
	- "@ " for persons
	- "! " for tweet
	- podcast, youtube, article, paper, thought
	- none for general Note
	- Have each type automatically load associated template
	- also valuable for searching ("& 2020" searches for books notes added in 2020)
- have templates with emoji tags 
	- tags for input type (see above)
	- tags for note lifecycle (new, fleshed out, complete but not fully interlinked, complete and tagged/linked)
- Todoist for task management (plugin links to obsidian but actual management would be outside of obsidian)
- habit tracking in daily note using tags (e.g. have tag for mood, daily reading, exercise, typing speed)
	- then access via dataview
	- make a Timeline note : [[supercharged links]] (plugin, renders tags from linked card in front of backlink on original card, works in conjunction with dataview to show)
	- 
- use tags for Note maturity and input type, not topics
- use links for MOC's ([[Map of Contents]])
- don't worry about folders


Dates in titles? 

How rigid of a structure (more rigid -> less rigid)
- Johnny Decimal
- folders
- Map Of Content networks
- free form

temporal component of what you're learning about
- on this day I processed this input -> what else did you learn about that day


Bryan's Goals (that I find relevant)
- external framework for thought
- Quick capture (in part due to his ADHD)
- notes on actual research papers
	- backlinking to component parts of a very complex paper
- technical notes (e.g. code snippets) are separate and on logseq

Quick Capture
- Todoist: add a task with a quick hotkey, has nlp, tags (
	- e.g. @obsidian <- tag
	- \#todo project
	- !!1 priority
	- comment, description)
	- also has a webapp, can be added to from the browser
	- api, can receive via email
	- mobile app, integrates with ios Shortcuts
- Note templates with tags 
	- nested tag system
		- any input get inbox emoji
			- different emojis for papers, notes, tweets, podcasts, articles 
				- in the case of notes, also has emoji for maturity (has it been fully fleshed out, fully backlinked)
				- in the case of papers, also has color emoji for has it been fully read, have all the citations of interest been fully explored
			- also emoji for Todoist items
			- map emoji, higher level notes as nodes to link other notes
			- people
		- Graph Groupings 
		- ![[Pasted image 20221121205127.png]]

Templater
- in conjunction with Templates
	- Templates/meta 
		- javascript logic to create a note with a different type of template depending on the first character of the title
		- ![[Pasted image 20221121204535.png]]
		- uses Kanban board plugin (could also use dataview queries) to identify Notes that need to get matured
		-  could use iCloud and shortcuts to create notes and send them to your inbox
		- use Obsidian mobile app?


Status Tracking, categorization, Taxonomy
- for tasks, he only tracks obsidian-related tasks in obsidian (e.g. Note curation, distillation, tending)
	- actual task management is in todoist
- tags utility items with gear icon (calendar, image manipulation, md reference, diagramming tools, etc.)

How to find things?
- 


tags vs links (associations traversal)
- tags = soft
	- don't overuse tags
		- use for status or large contextless groupings (seedling note, evergreen note or input type like books, articles etc.)
	- tags don't have unlinked mentions feature
- links = hard
	- shows intentIonal connection
	- use links as much as possible
	- Use links for MOC's (psychology, society, ADHD)
		- may need to manage MOC levels (e.g. Psychology > category of psychology)

temporal components --> Daily notes

What else?

Literature notes

Evergreen Notes

Moc's (emojis)

Maturing notes in the wild

Favorite Tools for PKM

Potentially have a input process (template) for each input type

Treat each individual markdown file as an atomic thought

Random note traversal for curating and polishing notes


#### Community plugins
- Codemirror matchbrackets.js community plugin for improving vim bindings
- Cycle through Panes ctrl-tab (seems to be included by default now)
- File explorer note count
- Metatable : more aesthetic meta data
- Mindmap : tree view of outlines
- Natural Language Dates
- Pane Relief (may by included now)
- ✅ Paste link in selection
- ✅ Relative line numbers ✅
- ✅ Recent Files
- Review : add a link to the current note to a daily note on a future date 
- Sliding Panes <- now built into obsidian as command : Toggle stacked panes
- ✅ Supercharged links
- Open Random Note
- Tag Wrangler : rename every instance of tag
- ✅ Templater :  check out github discussions for cool examples of usages
- Zoom : zooms into particular areas of a note, useful for outline like writing and input
- MathPix
- 



example: dataview query with templater code in it
shows links to all notes created on the same day of the current note
![[Pasted image 20221203201703.png]]

link to day prior and day next on each daily note
![[Pasted image 20221203201803.png]]