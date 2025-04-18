---
date: 2022-12-18
time: 16:42
note_maturity: 🌱
tags: 
---
# LLM AppIication Ideas

## Automotive QA:

- understanding search results in the automotive domain can be difficult because it requires a large amount of context
- using ChatGPT to resolve automotive questions might be more effective
- you may get good results by first gathering specific information from the user like model, make, year, mileage, then combining that with the user's actual query in a way that intelligently asks ChatGPT the question
- provide this as a service alongside ads potentially

To generalize the automotive idea, it might be the case that ChatGPT can be good for sourcing answers in domains where the individual doesn't have a lot of contextual knowledge. 

## LLM-powered fact checking app for articles

Start with just analysis of single claims using web search and LLM reasoning.
Layer on top of that analysis of whole article at once.

[[Drawing 2024-04-21 08.07.04.excalidraw]]

Planning
- generate summary of article
- generate supporting claims

Actions
- grade each supporting claim
- web search
- web search using domain specific trusted sources

Memory
- chat history and vector store?
	- vector store of trusted sources wikipedia?

Answer Generation
- Synthesize summary and analysis of supporting claims
- provide sources

## Memory Manager: your LLM-powered reminders and todo list

- remind me to stretch, exercise, study every day
- general rules for when to highlight what
	- time-of-day based reminders
- end of day notification of what has/hasn't been done
- summary of task list
- reminders for recently added or soon due todo's
- Core features:
	- habit building
	- task management

## Unit Searcher

build tools for tenants finding better options based on preferences
	- scraping available rentals
	- better ways to analyze tradeoffs in a more messy way
		- price
		- pet friendliness
		- proximity to dog parks, parks
		- proximity to hikes / parks
		- proximity to gyms}
		- proximity to workplace
		- proximity to restaurants, coffee shops
		- proximity to friends
		- bedrooms / baths / sq. footage
		- weighing all these things
		> *does llm provide any value or are these all just metrics we should build and present*
		- *is this a service we can sell or no*
	- maintenance education?

## PR builder and Code Reviewer:

1. build context
	1. from ticket- requirements
	2. from repo- project structure, relevant files
2. propose plan / architecture
3. implement planned steps

- each step should be collaborative
- each step should be update-able

>self feedback: how is this different from aider...

PR reviewer:
- within each repo, have a file that specifies how PR's should be reviewed
- provide that specification in-context when reviewing a PR
- provide tooling to easily add additional files to the context of a review if the changes require additional context



















# 🧭  Idea Compass
- West  (similar) 
[[Machine Learning]]
[[ChatGPT]]
[[DOM Chat App]]
- East (opposite)

- North (theme/question)

- South (what follows)
