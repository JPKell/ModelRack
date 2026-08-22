# CLI Standards

**Applies to:** `freeweight`, `loadcoach`, `ideapress`.
**Framework:** Typer (Click underneath) — typed signatures, generated help, shell completion.
**Rule:** every CLI command calls the same service layer as the web UI. Command bodies parse input,
call one service, render output. No business logic lives here.

---

## 1. Command shape

```text
<app> <noun> <verb> [ARGS] [OPTIONS]

freeweight run start --model ollama/qwen3.5:9b-q8_0 --suite native.performance
freeweight run list --status completed --limit 20
freeweight results export --run 01J9K2M --format json --output ./run.json
loadcoach job submit --task code.review --prompt-file review.txt
loadcoach route explain --task code.review
ideapress project create "Article on local inference"
ideapress stage run draft --project article-on-local-inference
```

Nouns are singular and match the domain vocabulary. Verbs come from a fixed list:
`list`, `show`, `create`, `start`, `stop`, `cancel`, `delete`, `export`, `import`, `run`, `submit`,
`explain`, `validate`, `init`, `check`.

Every application provides these top-level commands:

```text
<app> serve            start the web server (the default when the app is run with no arguments)
<app> health           dependency health, with --json
<app> version          version, API versions, schema versions
<app> config show|validate|init|path
<app> db upgrade|status|backup|restore|vacuum
<app> models list|show|refresh
<app> token create|list|revoke        (only meaningful when auth is configured)
<app> doctor           actionable diagnosis of a broken installation
```

---

## 2. Help and discoverability

* `--help` on every level; `-h` as an alias. Help shows: one-line purpose, usage, arguments,
  options with defaults and env-variable names, and at least one realistic example.
* `<app> --version` prints `<app> X.Y.Z (api v1, schemas benchmark.result 1.1)`, or a JSON object
  with `--json`.
* Errors that stem from a wrong invocation print the relevant usage line, not the whole help.
* Shell completion for bash/zsh/fish via `<app> --install-completion`.
* Command help must render correctly in an 80-column terminal.

---

## 3. Output

* **Human-readable by default.** Aligned columns, units, colour when the output is a TTY.
* `--json` on every command that returns data: a single JSON document on stdout, nothing else, and
  the exact same field names as the HTTP API. `--json` implies `--no-color` and `--quiet`.
* `--jsonl` where the output is a stream of records (samples, events, jobs).
* Colour is disabled automatically when stdout is not a TTY, when `NO_COLOR` is set, or when
  `--no-color` is given.
* Progress bars and spinners only on a TTY; in non-interactive mode, periodic one-line status
  updates instead. `--quiet` suppresses everything but errors and the final result.
* stdout carries data; **stderr carries logs, warnings and progress**. Piping `--json` into `jq`
  must never require filtering.
* Long-running commands stream events as they happen (the same events the web UI receives), so a
  CLI run is as observable as a browser run.

---

## 4. Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Generic failure (unexpected) |
| 2 | Usage error — bad arguments, unknown command |
| 3 | Configuration error — invalid config, unsafe combination, missing prompt pack |
| 4 | Dependency unavailable — provider down, database needs migration, peer unreachable |
| 5 | Operation failed — the run/job/stage executed and did not succeed |
| 6 | Cancelled — by the user or by a signal |
| 7 | Resource exhausted — insufficient VRAM/RAM/disk, queue full |

Codes are stable and part of the public contract; scripts branch on them. Each is tested.

---

## 5. Non-interactive operation

* Every command must be runnable non-interactively. Anything that would prompt takes an explicit
  flag (`--yes`, `--force`, `--name …`).
* When stdin is not a TTY, prompts are errors (exit 2) naming the flag that would have answered them.
* No command requires a browser, a display, or a running server other than its own dependencies.
* `--dry-run` on every destructive or expensive command, printing exactly what would happen.
* Signals: `SIGINT`/`SIGTERM` cancel cleanly — the in-flight unit is marked cancelled, partial
  results are preserved, the database is left consistent, exit code 6. A second signal aborts hard.

---

## 6. Local vs. remote execution

Some commands need a *running* server (live queue state, cancellation of another process's job).
Each command declares its mode:

| Mode | Behaviour |
|---|---|
| **Local** | Runs the service layer in-process against the configured database. Works with no server running. |
| **Client** | Talks to the configured server over HTTP; fails with exit 4 and the URL if it is not reachable. |
| **Either** | Uses the server when one is reachable, otherwise runs locally; `--local`/`--remote` force the choice, and the output states which was used. |

The mode is documented in each command's help text. Commands that mutate state which a running
server also owns (starting a run, cancelling a job) are **client** mode when a server is up, to
avoid two writers racing on the same database.

---

## 7. Arguments and options

* Long options are `--kebab-case`; short options exist only for the frequent ones
  (`-o/--output`, `-f/--format`, `-v/--verbose`, `-q/--quiet`, `-y/--yes`).
* Booleans are `--flag` / `--no-flag` pairs, never `--flag=true`.
* Values that could be large (prompts, documents) accept `--x-file PATH` or `-` for stdin, in
  addition to the inline form.
* Repeatable options for lists: `--model A --model B`.
* Every option maps to a config key and/or an environment variable, and the help shows it.
* Model references accept the canonical ID (`ollama/qwen3.5:9b-q8_0`) or an unambiguous prefix; an
  ambiguous prefix is an error that lists the candidates.
* IDs accept an unambiguous prefix (`01J9K2M` for a full ULID) everywhere.

---

## 8. Errors

```text
Error: provider unavailable (PROVIDER_UNAVAILABLE)

  Ollama did not respond at http://127.0.0.1:11434 (connection refused).

  Try:
    • Start it:            ollama serve
    • Check the setting:   freeweight config show | grep provider
    • Use another host:    freeweight --provider-url http://…

  Request ID: 01J9K2M4P7Q8R9S0T1U2V3W4X5
```

* Every error names: what failed, the stable `code`, why, and at least one next step.
* Stack traces are hidden unless `--verbose`/`--debug`; the request ID always links to the full log.
* Never print a secret, and never print a path outside the data root.
* With `--json`, errors are the standard error envelope on **stdout** and a non-zero exit code, so
  scripts can parse them.

---

## 9. Configuration inspection

```bash
<app> config show              # effective config with the source of every value
<app> config show --json
<app> config validate          # exit 0/3 without starting anything
<app> config path              # resolved config file location
<app> config init              # write a commented example file
```

Secrets always render as `********`.

---

## 10. Health and diagnostics

```bash
<app> health          # per-component status, worst-first, exit 0 (ok/degraded) or 4 (unavailable)
<app> health --json
<app> doctor          # deeper checks with fixes: DB revision, provider reachability, GPU tooling,
                      # sandbox tier, permissions on the data root, prompt pack validity,
                      # port availability, peer application compatibility
```

`doctor` prints a checklist with ✓ / ! / ✗ and a suggested command for each failure. It is the first
thing a support conversation asks for.

---

## 11. Scriptability

* Stable field names between `--json` output and the HTTP API — one vocabulary.
* Stable exit codes.
* No interactive fallback.
* Idempotent where the operation permits it (`db upgrade` on a current database is a no-op with exit 0).
* IDs are printed on creation so a script can chain commands:

```bash
run_id=$(freeweight run start --model … --suite … --json | jq -r .run_id)
freeweight run wait "$run_id" --timeout 3600
freeweight results export --run "$run_id" --format json --output result.json
```

* `<app> <noun> wait <id>` exists wherever an operation is asynchronous, with `--timeout` and an
  exit code reflecting the terminal state.

---

## 12. Startup performance

`--help` and `--version` must not import SQLAlchemy, FastAPI, httpx or Jinja2. Heavy imports are
deferred into the command body. Budgets: `--help` ≤ 250 ms; a simple command ≤ 500 ms end to end
([Performance Targets](../architecture/performance-targets.md) §3.6). A test asserts the imported
module set for `--help`.

---

## 13. Testing

Every command is tested for: success path, `--json` shape, each documented exit code, missing
required argument, non-interactive behaviour with a piped stdin, `--dry-run`, cancellation by
signal, and help rendering at 80 columns. CLI tests use Typer's runner against a fake-provider
composition root — no server and no model required.
