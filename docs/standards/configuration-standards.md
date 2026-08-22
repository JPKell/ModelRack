# Configuration Standards

**Applies to:** all three applications. Shared packages take configuration as constructor
arguments and never read files or environment variables themselves.

---

## 1. Precedence

Later sources override earlier ones, field by field:

```text
1. built-in defaults        (in code, safe, local-only, always valid)
        ↓
2. configuration file       (TOML; ~/.config/<app>/config.toml unless --config is given)
        ↓
3. environment variables    (<APP>_ prefixed)
        ↓
4. CLI arguments            (highest)
```

Rules:

* Overriding is per **leaf field**, not per section. Setting `FREEWEIGHT_SERVER__PORT` does not
  discard the rest of `[server]`.
* Precedence is identical for the web server, the CLI, and any background worker — they load the
  same settings object through the same function.
* `<app> config show` prints the effective configuration **with the source of every value**, which
  is how precedence questions get answered in practice:

```text
server.host          127.0.0.1   (default)
server.port          8790        (env FREEWEIGHT_SERVER__PORT)
storage.database_url sqlite:///… (file ~/.config/freeweight/config.toml)
telemetry.interval_ms 500        (cli --telemetry-interval-ms)
provider.api_key     ********    (env FREEWEIGHT_PROVIDER__API_KEY)
```

### 1.1 The other precedence chain (do not confuse them)

Benchmark and job **execution parameters** resolve on a different axis, *inside* an already-loaded
application configuration:

```text
application defaults → suite defaults → test/task defaults → saved user settings → run/job overrides
```

This chain governs temperature, context size, repetitions, timeouts and similar per-execution
values, and its resolved output is frozen into the run/job record so history never changes when
global settings change later. §1 governs how the *application* is configured; this chain governs
how a *unit of work* is parameterized. They never merge.

---

## 2. Format and location

* **TOML** ([ADR-0019](../adr/0019-python-baseline-and-config-format.md)): stdlib `tomllib`,
  comment-friendly, unambiguous types, no dependency.
* Default paths (XDG-respecting, with the documented fallback shown):

```text
config   $XDG_CONFIG_HOME/<app>/config.toml    → ~/.config/<app>/config.toml
data     $XDG_DATA_HOME/<app>/                 → ~/.local/share/<app>/
state    $XDG_STATE_HOME/<app>/                → ~/.local/state/<app>/
```

* `--config PATH` overrides the file location entirely. `<app> config init` writes a fully
  commented example file; `<app> config path` prints the resolved location.
* A project-local `./<app>.toml` is used when present and no `--config` is given, so a directory can
  carry its own settings; `config show` names which file won.
* Missing config file is normal, not an error: defaults apply.

---

## 3. Environment variables

* Prefix per application: `FREEWEIGHT_`, `LOADCOACH_`, `IDEAPRESS_`.
* Nesting uses a double underscore: `[server].port` ⇒ `FREEWEIGHT_SERVER__PORT`.
* Values are parsed with the field's declared type; `true/false/1/0/yes/no` for booleans; comma
  separation for lists.
* An unparseable value is a startup error naming the variable, the value and the expected type — it
  is never silently ignored or coerced.
* Reserved: `<APP>_CONFIG` (config file path), `<APP>_DATA_DIR`, `<APP>_LOG_LEVEL`,
  `NO_COLOR`, `XDG_*`.

---

## 4. Schema and validation

Configuration is a typed model, validated at startup, before anything opens a socket or a database:

```python
class ServerSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1, le=65535)
    allow_lan_exposure: bool = False
    allowed_hosts: tuple[str, ...] = ()      # required when host is not loopback (ADR-0026)
    request_timeout_seconds: float = Field(default=120.0, gt=0)

class Settings(BaseSettings):
    server: ServerSettings = ServerSettings()
    storage: StorageSettings = StorageSettings()
    provider: ProviderSettings = ProviderSettings()
    telemetry: TelemetrySettings = TelemetrySettings()
    logging: LoggingSettings = LoggingSettings()
    model_config = SettingsConfigDict(
        env_prefix="FREEWEIGHT_", env_nested_delimiter="__", extra="forbid"
    )
```

* **Unknown keys are rejected** with the key name and the closest valid key suggested. A silently
  ignored setting is worse than an error.
* Cross-field rules are validated too, and refusal is the correct behaviour for unsafe combinations:
  * non-loopback `host` without configured tokens ⇒ `INSECURE_BINDING`
  * non-loopback `host` without `server.allowed_hosts` ⇒ `INSECURE_BINDING` (a service reachable under
    any hostname cannot be protected against DNS rebinding — [ADR-0026](../adr/0026-local-http-hardening.md))
  * `host = "0.0.0.0"` without `allow_lan_exposure = true` ⇒ refuse
  * CORS enabled without configured tokens ⇒ refuse (it withdraws the JSON API's CSRF exemption)
  * remote provider configured without `providers.allow_remote = true` ⇒ refuse
  * `storage.database_url` pointing outside the data root without `--allow-external-db` ⇒ refuse
  * `queue.lease_seconds` not greater than three renewal intervals plus slack ⇒ refuse (LoadCoach;
    [ADR-0029 §4](../adr/0029-queue-mechanics.md))
* Validation failure exits with code 3 and a message naming file, key, value and expectation.
* `<app> config validate [--file PATH]` runs the same validation without starting the service, so it
  can be used in CI and in deployment scripts.

---

## 5. Defaults

Defaults must produce a working, safe, local installation with **no configuration file at all**:

| Setting | Default | Why |
|---|---|---|
| `server.host` | `127.0.0.1` | Local-first; never exposed by accident |
| `server.port` | 8765 / 8766 / 8767 | Distinct per application |
| `server.allow_lan_exposure` | `false` | Exposure is a deliberate act |
| `storage.database_url` | `sqlite:///<data_dir>/<app>.sqlite3` | Zero configuration |
| `storage.auto_migrate` | `true` for SQLite, `false` for PostgreSQL — the default is **dialect-dependent**, resolved from `database_url`, and the value shown by `config show` names which applied | Shared databases are upgraded deliberately |
| `server.allowed_hosts` | empty (loopback allowlist is implicit) | Required only for a non-loopback bind |
| `provider.kind` | `ollama` | The suite's first-class provider |
| `provider.base_url` | `http://127.0.0.1:11434` | Ollama's default |
| `providers.allow_remote` | `false` | No content leaves the machine unless asked |
| `telemetry.interval_ms` | 1000 | Readable without measurable overhead |
| `telemetry.persist_during_runs` | `true` | Needed for energy and peak-VRAM metrics |
| `logging.level` | `INFO` | |
| `logging.format` | `text` on a TTY, `json` otherwise | Humans and machines both served |
| `logging.include_content` | `false` | Prompts and outputs are not log material |
| `auth.tokens` | empty | No auth needed on loopback |

---

## 6. Secrets

* Config files never contain live secrets. They may contain a **reference**:

```toml
[provider]
kind = "openai_compatible"
base_url = "https://api.example.com/v1"
api_key_env = "EXAMPLE_API_KEY"        # name of the variable
# or
api_key_file = "~/.config/freeweight/example.key"   # mode must be 0600
```

* Resolution order for a secret: `*_env` → `*_file` → OS keyring (when available). A `*_file` whose
  mode is group- or world-readable is refused at startup.
* Secrets never appear in `config show`, logs, errors, exports or API responses (see
  [Security Standards](security-standards.md) §8).

---

## 7. Runtime-changeable settings

Some settings are editable through the UI and stored in the application database rather than the
config file (theme, table column choices, telemetry interval, default benchmark parameters, saved
routing weights).

Rules:

* Database-backed settings sit **between** file and environment in precedence:
  `defaults → file → database → env → CLI`. This is the one deviation from §1, it exists only for
  settings a UI can change, and it is documented on each such setting in the app's spec.
* Anything security-relevant (bind address, exposure flag, auth tokens, remote-provider allowance,
  database URL, data root) is **file/env/CLI only** and never editable from the UI.
* `config show` marks database-sourced values as `(database)`.

---

## 8. Configuration reference documentation

Each application publishes `docs/configuration.md` in its own repository, generated from the
settings model so it cannot drift, listing per field: key path, env variable, type, default, valid
range, whether it is runtime-changeable, security implications, and an example. A CI test fails when
the generated document differs from the committed one.

---

## 9. Testing

* Precedence: default → file → env → CLI, asserted for a representative field at each level.
* Partial override does not discard sibling fields.
* Unknown key rejected with a helpful message.
* Type/range violation rejected with the field named.
* Every unsafe combination in §4 refuses to start.
* `config show` redacts every secret-shaped value.
* Every default in §5 is asserted, so a default cannot drift unnoticed.
* Settings load in a temporary XDG root; no test reads or writes the developer's real config.
