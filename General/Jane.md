---
date: 2025-01-04
time: 11:11
note_maturity: 🌱
tags:
  - project
---

# Jane

Todos:
- set and use backend_host as global env variable
	- validation loop and creation of evaluations / samples?
- Clean up interrupt pattern (run agent vs resume agent)
- add seed sqlite db to source
- add langchain

## Architecture and Design

### CRUD

- Models:
	- messages
	- todos
- Potential Models:
	- summary of day's conversation (short term memory summarization)
	- goals
	- daily note including updates relevant to a given goal

## Ideas:

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

- Langgraph Agent Swarm
	- essentially uses Command between agent calls to keep 
	- potentially use to implement a validator agent
	- potentially use a separate validator agent in a lambda
		- each subagent and the top agent have a `validator_params` state variable
		- at each agent handoff, send the validator params to the validator agent in a separate lambda (to avoid increasing latency)
		- alternatively, wait for the validator output at each handoff and kick it back if the validator decides it was wrong
			- is this just overcomplicating? what benefit do you have of using a validator agent rather than building this into each graph
			- params: messages, input (prompt, available tools), output (tool, tool params),  evaluator_prompt_1, evaluator_prompt_2...

### Task Tracking and Report Generation

- Build a tool to start writing tasks completed, workouts done, etc. to a daily note. 
- Note writer should use structured output to ensure format is respected. 
- Create a report generator that summarizes tasks completed and creates a report recommending my promotion. 
- Create a report generator that reports on recent activity levels by desired metrics. 
- connect goals to tasks in reports

### Interview Mode / Data Collection Mode

- Build an interview-like mode that proactively asks for data from your day. 
- ask about exercise, learnings, etc.
- run as a scheduled job?

### Chat Thread Management and Persistence

- store each thread in the db?
- store summary of thread in the db?
- store record of actions taken

### Memory

- long term memories
- store long term memories after action confirmed
	- or when a data response has been liked?
- how to resurface memories in future prompts?
	- simple prompt injection with vdb similarity?
	- tool calling to fetch similar past memories when uncertain?

### Evals 

- investigate `deepeval` 
- investigate langchain trajectory evals

### MCP

What MCP tools could be useful to Jane?
