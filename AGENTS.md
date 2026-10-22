# kagent

Development of an Autonomous ML kaggler

## User Clarifications

### Interviewing the developer about how to do a task:
When asked for a large piece of work which seems vague or needs clarification, please interview me in detail using the AskUserQuestionTool about literally anything: technical implementation, UI & UX, concerns, tradeoffs, etc. but make sure the questions are not obvious. Be very in-depth and continue interviewing me continually until it's complete, then write the learnings to README.md


## Coding guidelines and philosophy

- You should generate code that is simple and redable, avoid unnecesary abstractions and complexity. This is a research codebase so we want to be mantainable and readable.
- Avoid overly defensive coding, no need for a lot of `try, except` patterns, I want the code to fail is something is wrong so that i can fix it.
- Do not add demo-only flags or placeholder CLI options that gate real functionality (e.g., `--run` just to toggle execution); scripts should run their main logic directly.
- Adhere to python 3.12+ conventions