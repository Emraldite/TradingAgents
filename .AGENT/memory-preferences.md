# User Preferences

- Be direct and avoid filler.
- Briefly explain non-obvious decisions.
- Keep solutions simple and do not over-engineer.
- Handle errors explicitly.
- Do not add dependencies without confirming first.
- Use `uv` for dependency management and never use bare `pip install`.
- Mobile-first design when working on frontend tasks.
- Do not start a dev server unless explicitly asked.
- Never commit on the user's behalf; staging only is allowed.
- Prefer Google/Gemini free-tier or other low-cost LLM usage when running agent analysis.
- Prefer Oracle VM deployment through git clone / git updates, with local logs, caches, DBs, venvs, and secrets excluded by `.gitignore`.
