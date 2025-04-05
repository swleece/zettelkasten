---
date: 2025-01-04
time: 11:11
note_maturity: 🌱
tags:
  - project
---

# Jane

Todos:
- start new conversation automatically after CRUD operation completion
	- `action_executed`
- set and use backend_host as global env variable
- validation loop and creation of evaluations / samples?

## Architecture and Design

### CRUD

- Models:
		- messages
	- todos
- Potential Models:
	- summary of day's conversation (short term memory summarization)
	- goals
	- daily note including updates relevant to a given goal

### Conversational Data Collection

- UX Considerations and Strategy:
	- ask the user before launching into data collection questions
		- make it clear what mode of data entry the chat bot is in and what data it is accepting, either via UI elements or via chat message
	- do data collection sessions based on a Goal? 
		- i.e. step through each goal and get a daily update on it
		- ask more detailed questions
	- provide the last 2 weeks? of updates as context so that the llm can provide a more contextual, personalized question/message
- Idea:
	- run a scribe_recorder agent in parallel for every message that tries to extract an update for each message when not interviewing user. Templatize the prompt to include the user's existing goals (fetched from db)
	- 


## Ideas:

### Advanced context management

- advanced context management:
	- create subagents in such a way that they are very composable and can be mixed and matched
	- teacher pattern subagent to bootstrap my own computer science
		- what would this mean? what are my weaknesses and how could I get those to be challenged?
		- pair programmer pattern? (without dupIication of aider / copilot etc. )
		- explain examples of data structures and algorithms?
		- explain architecture patterns?

- think more about context management
	- how long can I just rely on messages history
	- how to get agents to smartly pull in context as needed
	- how to model context and use State objects smartly
	- how much guardrails, how much freedom

### Goals Manager

- Goals CRUD
- connect goals to tasks in reports

### Task Tracking and Report Generation

- Build a tool to start writing tasks completed, workouts done, etc. to a daily note. 
- Note writer should use structured output to ensure format is respected. 
- Create a report generator that summarizes tasks completed and creates a report recommending my promotion. 
- Create a report generator that reports on recent activity levels by desired metrics. 

### Interview Mode / Data Collection Mode

- Build an interview-like mode that proactively asks for data from your day. 
- ask about exercise, learnings, etc.

### Thread management and Persistence

- store each thread in the db?
- store summary of thread in the db?
- store record of actions taken
- 

### Evals 

- investigate `deepeval` 



## Brainstorming

I'm building a conversational AI chat bot powered by an LLM to track my habits and goals. Can you help me brainstorm design ideas for an input mode where the chat bot will proactively ask the user for updates from the day?

Requirements:

1. the bot should only ask for updates if an update hasn't already been provided in the chat history for the current day
2. the bot should proactively reach out to the user in the evening if no updates have been given

Some of my ideas currently:

- my app does not currently have a way to run scheduled jobs. I could build a way to run a scheduled job that initiates a conversation in case 2