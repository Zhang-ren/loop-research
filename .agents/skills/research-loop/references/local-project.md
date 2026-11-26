# Local experiment onboarding

Use a single reusable entry Skill with the same local CLI across projects. Put each experiment's specification and study state in its own run directory. Prefer a path such as `.loop-research/<study-name>` under the user's project and keep generated run directories out of the source snapshot.

`prepare-study` reads `PROJECT/experiment.json` or the file passed through `--spec`, then asks a Codex role to identify missing documented inputs. In a terminal it prompts for answers. `--non-interactive` returns structured issues, current configuration and evidence sources for an agent to present. `--answers` takes a JSON object such as `{"metrics":[{"name":"accuracy","direction":"max","threshold":0.9}],"command_experiment.command":["python3","evaluate.py"]}`. Partial replies persist in `RUN/preparation.json`. Use the same run directory on each retry. Changing the budget, command or evaluation protocol can require renewed confirmation.

The command adapter pins a Git commit. Its candidate scope is a JSON parameter object with declared choices. A trusted command can call an existing Python, R, shell or compatible pipeline entrypoint in a clean checkout. The checkout uses a Git archive. Commands requiring repository metadata, such as `dvc exp run`, need a compatible adapter. Configure the necessary runtime via an absolute interpreter/tool path when the system PATH cannot find it. The wrapper reads:

- `LOOP_PARAMETERS` for the JSON parameter object.
- `LOOP_SPLIT_MANIFEST` for train IDs and the current evaluation IDs.
- `LOOP_SPLIT` for `validation` or `test`, and `LOOP_SEED` for a fixed seed.
- `LOOP_METRICS` for writing a JSON object such as `{"metrics":{"accuracy":0.9}}`. Optional `predictions` and `scores` must follow the selected split's ID order.

Commit the evaluator, experiment assets and a split-manifest JSON with disjoint `train`, `validation` and `test` lists. The command must use the supplied split and measure candidates on the same workload. The project must supply the experiment's data dependencies. The runner validates the declared IDs and pins committed content. It trusts the supplied execution command and evaluator to honor that contract.

Set optional `environment.committed_files`, `environment.local_paths`, `environment.python_modules`, `environment.programs` or `environment.gpu_count` for requirements that the command alone cannot expose. Preparation checks those requirements, validates the split, and writes a commit-pinned `RUN/study-spec.json` once ready. Add `--start` to recheck the environment, measure the validation baseline and start candidate optimization after success. A failed baseline returns to `awaiting_input`; see `status RUN` and `RUN/baseline-logs`. After committing a fix to the evaluator, supply the new `command_experiment.revision` in `--answers` and confirm the revised protocol. With `independent_review` enabled, a disagreement creates a review reference and suspends selection. Use `resolve-review` only with the user's actual decision and rationale. With `followup_experiments` enabled, combinations and numeric boundary probes use the same candidate budget. An exact losing candidate can be excluded using an evidence-bound `record-refutation` decision.

Different study roots, including separate runs of the same source repository, have distinct IDs and separate memory. Reopen the same root to continue with approved experience. Keep generated Skills under that root. The reusable onboarding Skill is the only component intended for shared discovery.
