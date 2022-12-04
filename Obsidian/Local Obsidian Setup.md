---
tags: obsidian, local_environment, productivity, pkm
---

#TODO create templates for daily notes, workout notes(or should that just be part of daily?)


### Hotkeys

| Command          | Action                                    |
| ---------------- | ----------------------------------------- |
| cmd-L            | create Task                               |
| cmd-;            | mark Task done                            |
| pipe, tab, enter | create table using Advanced Tables plugin |
|                  |                                           |


### Community Plugins

- Advanced Tables
- [Dataview](https://blacksmithgu.github.io/obsidian-dataview/) 
	- Dataview is a live index and query engine over your knowledge base. You can associate _data_ (like tags, dates, snippets, numbers, and so on) with your markdown pages, and then _query_ (like filter, sort, transform) this data. This is a simple but powerful idea:
		-   Track sleep schedules and habits by recording them in daily notes, and automatically create weekly tables of your sleep schedule.
		-   Automatically collect links to books in your notes, and render them all sorted by rating.
		-   Automatically collect pages annotated with a given date, showing them in your daily note or elsewhere.
		-   Find pages with no tags for follow-up, or show pretty views of specifically-tagged pages.
		-   Create dynamic views which show upcoming birthdays or events, annotated with notes.
	- Dataview is highly generic and high performance, scaling up to hundreds of thousands of annotated notes without issue. If the built in [query language](https://blacksmithgu.github.io/obsidian-dataview/query/queries/) is insufficient for your purpose, you can run arbitrary JavaScript against the [dataview API](https://blacksmithgu.github.io/obsidian-dataview/api/intro/).
	- queryable data: frontmatter (metadata), Inline Fields(e.g. "\[rating:: 9\]"), implicit (e.g. file.cday <-created date, file.outlinks etc.)
	- metadata ideas: last-reviewed, tags, type, ...
	```dataviewDemoFormat
	TABLE|LIST|TASK <field> [AS "Column Name"], <field>, ..., <field> 
	FROM <source>
	WHERE <expression>
	SORT <expression> [ASC/DESC]
	... other data commands
	```
	```dataviewDemoExample
	TABLE
	  time-played AS "Time Played",
	  length AS "Length",
	  rating AS "Rating"
	FROM "games"
	SORT rating DESC
	```
- Calendar
- Kanban
- Tasks
- Editor Syntax Highlights
- [Templater](https://silentvoid13.github.io/Templater/)
- supercharged links
- 


### Plugins
- daily note

### Settings
- vim mode
- 