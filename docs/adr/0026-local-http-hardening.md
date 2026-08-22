# ADR-0026 — Local HTTP hardening: Host validation, CSRF and outbound fetch

**Status:** Accepted (2026-08-21)
**Extends:** [ADR-0014](0014-authentication-strategy.md), [Security Standards](../standards/security-standards.md).

## Context

ADR-0014's default is deliberate and correct: loopback bind, no credentials, the OS user boundary as
the security boundary. It rests on the premise that only local code can reach the port.

That premise has two well-known holes, and the audit found neither addressed:

1. **DNS rebinding.** A web page the user visits can resolve an attacker-controlled hostname to
   `127.0.0.1` and then issue same-origin requests to it. The browser's origin check passes, because
   the origin is the attacker's hostname. Every unauthenticated loopback service in the suite is
   reachable this way — including LoadCoach, which will execute inference, and IdeaPress, which holds
   the user's projects. The defence is a `Host` header allowlist, and nothing in the documentation
   required one.
2. **CSRF on HTML forms.** Security Standards §1 asserted "CSRF protection on state-changing UI
   forms" without naming a mechanism, and §14's security-test list contained no CSRF test. The JSON
   API is incidentally protected — it requires `Content-Type: application/json`, rejects other media
   types with 415, and CORS is disabled, so a cross-origin `fetch` fails preflight — but a
   form-encoded POST is not preflighted at all.

Separately, `POST /api/v1/evidence/import` accepts `{"url": "http://…"}` and makes LoadCoach fetch it.
On the default loopback deployment that endpoint needs no credential, so any local process — or a page
that has rebound DNS — can direct the application to fetch an arbitrary URL and parse the result. That
is a server-side request forgery primitive with LoadCoach's network position, and it is not covered by
`providers.allow_remote`, which governs model providers only.

None of this changes the threat model. It closes the gap between the model and the implementation.

## Decision

### 1. Host header allowlist

Every application validates the `Host` header on every request, before routing:

* Loopback bind: accept `localhost`, `127.0.0.1`, `[::1]` and the literal bound address, each
  optionally with the configured port. Anything else → **421 Misdirected Request**, logged at WARNING
  with the presented value and the request ID.
* Non-loopback bind: accept only the hosts in `server.allowed_hosts`, which is **required** when
  `server.host` is not loopback — a third member of the existing bind/token/acknowledgement refusal
  set in [Configuration Standards §4](../standards/configuration-standards.md).
* The check is a MirrorWall middleware so all three applications behave identically, and it runs
  before authentication so an unauthenticated rebinding attempt never reaches a route.

### 2. CSRF, with a named mechanism

* **HTML form routes** (`POST` from a rendered page) carry a double-submit token: a `__Host-`-prefixed
  `SameSite=Strict` cookie plus a hidden field, compared with `hmac.compare_digest`. Mismatch or
  absence → 403 `CSRF_FAILED`.
* **The JSON API is exempt**, on stated grounds rather than by omission: it accepts only
  `application/json` (415 otherwise), which cannot be produced by a cross-origin form and forces a
  CORS preflight that fails while CORS is disabled. If CORS is ever enabled, the API's exemption is
  withdrawn and bearer tokens become mandatory — enforced at startup.
* Both facts are tested: a forged form post is rejected; a cross-origin JSON post is rejected; a
  same-origin form post with a valid token succeeds.

### 3. Application-initiated outbound fetches

Any fetch whose URL comes from a request body — today only `POST /evidence/import` — obeys:

* Scheme in `{http, https}` only.
* Host in `evidence.allowed_source_hosts`, **default `["127.0.0.1", "localhost", "::1"]`**. A remote
  FreeWeight must be listed explicitly, which is the same "exposure is a deliberate act" posture the
  suite already takes for binding and for remote providers.
* Literal-IP destinations in link-local (`169.254.0.0/16`, `fe80::/10`), and any address that resolves
  into them, are refused unconditionally — the cloud metadata range is the classic target.
* Redirects are not followed across a host change; total redirects capped at 3.
* A response size cap (the import limit, 128 MiB) enforced during streaming, a connect timeout and a
  read timeout, and `Content-Type` verified before parsing.
* No credentials are attached unless the source is configured with one (§4), and none are ever
  forwarded across a redirect.
* The endpoint remains `admin`-scoped when authentication is on; on loopback the host allowlist and
  the default of "loopback only" are what stand in its place.

### 4. Authenticated evidence sources

`[evidence]` gains a credential alongside `freeweight_url`, because FreeWeight bound non-loopback
requires a bearer token and LoadCoach previously had no way to supply one:

```toml
[evidence]
freeweight_url          = ""
freeweight_api_key_env  = ""     # or freeweight_api_key_file, per Configuration Standards §6
allowed_source_hosts    = ["127.0.0.1", "localhost", "::1"]
```

The credential resolves through the ordinary secret chain, is redacted everywhere, and needs only
`read` scope on FreeWeight — whose `/evidence` endpoints already require nothing more.

### 5. `GET /api/v1/version` is unauthenticated

Version negotiation happens before a client can know whether its credential is right. If `/version`
required a scope, a bad token and an incompatible API would be indistinguishable. It returns only
version metadata — no counts, no names, no configuration — and is exempt from authentication in all
three applications. `/health` continues to require `read` when authentication is on, because its
component detail is operational information.

## Alternatives considered

**Rely on loopback alone.** The status quo. Rejected: DNS rebinding defeats it, it is a decade-old
technique, and the assets behind these ports are a GPU and the user's private work.

**Require authentication on loopback.** Rejected again for ADR-0014's reasons — it destroys
zero-configuration startup. Host validation buys the same protection against the actual attack for
none of the friction.

**Origin-header checking instead of Host.** Rejected as the primary control: `Origin` is absent on
many legitimate requests (`curl`, the CLI, server-side clients) and present-but-attacker-controlled in
exactly the rebinding case. `Host` is what the attack must forge and cannot.

**A CSRF token on the JSON API too.** Rejected: it breaks `curl` and every scripted client for a
threat the content-type and CORS posture already closes, and the exemption is now stated with its
reasoning and its withdrawal condition rather than left implicit.

**Drop the `{"url": …}` import form and accept only uploads.** Genuinely tempting, and it removes the
SSRF surface entirely. Rejected because pulling from a FreeWeight on the same machine is the ordinary
path and the one that makes the two applications feel connected; the allowlist keeps the default
identical in reach to an upload-only design while leaving the LAN case configurable.

## Consequences

*Positive.* The local-first default becomes defensible against the attack that actually targets it.
The SSRF surface is reduced to "hosts the user named". LoadCoach can finally authenticate to a
FreeWeight that requires it — a deployment the architecture claimed to support and could not.
Version negotiation works before credentials are established.

*Negative.* A user reaching the UI through a hostname other than `localhost` (a `/etc/hosts` alias, a
container name) now needs `server.allowed_hosts`. The 421 response names the header it saw and the
setting that would accept it.

*Negative.* Four more startup validations and a middleware in the request path. The middleware is a
string comparison; the budget is unaffected.

## Revisit when

The suite gains a genuine multi-user deployment with an identity provider (ADR-0014's own trigger), at
which point session handling replaces the double-submit token and the CSRF section is rewritten
around it.
