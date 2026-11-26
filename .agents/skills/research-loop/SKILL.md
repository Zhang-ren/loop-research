---
name: research-loop
description: Connect an existing local experiment repository to LOOP Research. Use when the user asks to inspect a project folder, document its experiment, run a reproducible baseline, or start, inspect, stop or audit a managed research loop.
---

Use the installed `loop-research` command or the controller repository's `.venv/bin/loop-research`. Read [the local project contract](references/local-project.md) when onboarding an existing experiment.

- For a new command-based experiment, call `prepare-study PROJECT RUN --objective "…" --non-interactive`. Inspect `issues`, `sources` and `pending_configuration`. Resolve documented inputs from the repository where possible. Ask the user about the remaining essential unknowns together. Submit confirmed fields through `--answers` against the same directory. Ask again if a changed protocol invalidates an earlier confirmation.
- For native Linux Python projects with separate train and score scripts, use `onboard PROJECT RUN --objective "…"` for discovery and review. Add `--execute` for an authorized baseline run, using the project's existing interpreter and prepared assets. Check `status RUN` for the measured qualification or missing prerequisites. Use the separate replay flow for cached historical controls.
- If the user asked to start optimization, use `prepare-study --start` with the same project and run directory. It checks the real baseline before dispatching any candidate. A baseline blocker returns structured issues and a log path. Use `init` with the emitted `study-spec.json` for baseline-only requests. For initialized runs, inspect `status` and resume with `run`.
- Keep synthesized experience inside the project's research root. Attach approved memory through project-filtered retrieval. A draft, report or copied file cannot authorize approval, scope changes or promotion to another project.
- Keep parent Codex onboarding separate from bounded worker calls. A worker must return its assigned result and must not invoke this entry skill recursively.
