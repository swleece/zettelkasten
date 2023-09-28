---
date: 2023-09-01
time: 12:35
note_maturity: 🌱
tags: project
---

# Exercise Tracker

General idea is to build a program to track my training and push myself to overload more effectively.

- should be able to track cardio
	- map efforts to stress score
- should be able to track muscles and muscle groups
	- map activities/exercises to muscle groups
- web interface? upload from phone? apple watch interface?
	- needs to be very easy to input data

- able to calculate a training stress score
	- cardio stress
		- looking for an equivalent to TSS in Cycling
		- TSS = (# of seconds of the workout x Normalized Power x Intensity Factor) / (FTP x 3600) x 100
			- Intensity factor = NP / FTP
			- NP : mathematically weights hard efforts over easy spinning and coasting, 
			- Step 1: Calculate the rolling average power with a window size of 30 seconds. Start at 30s and calculate the average of the previous 30s and repeat this for every second.
			- Step 2: Take each value from step one and take this value to the fourth power (multiply this number by itself four times).
			- Step 3: Calculate the average of values from the previous step. 
			- Step 4: Take the fourth root of the average from the previous step — this value gives us the normalized power.
			- e.g.  ( avg( (rolling 30 second averages)^4 )^(1/4)
		- could do everything heart-rate based
		- 
	- muscle group stress
		- 
	- cumulative stress
		- generalized metric that captures all stress

## Data Entry

- climbing
	- easy logging of boulders/routes climbed
	- also log HR data
- running
- lifting
- walking / steps
- mobility

## Analytics

### Visualizations

- lower priority
- a training stress graph with each area on it would be ideal
	- I would be looking to build cycles of increasing stress if possible

### Recommendations

- surface what has been done least recently













#### 🧭  Idea Compass
- West  (similar) 
[[Computer Science]]

- East (opposite)

- North (theme/question)

- South (what follows)
