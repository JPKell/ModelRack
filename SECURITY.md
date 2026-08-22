# Security Policy

ModelRack is part of the Local AI Suite, whose default posture is local-first: it binds to
`127.0.0.1` by default, requires no credentials on loopback, and makes no outbound network
connections other than to a configured model provider. See
`docs/standards/security-standards.md` for the full trust-boundary model and
`docs/adr/0014-authentication-strategy.md` and `docs/adr/0026-local-http-hardening.md` for the
authentication and LAN-exposure design this component follows.

## Reporting a vulnerability

Please do not open a public issue for a suspected security vulnerability.

Instead, report it privately to the maintainer with:

* A description of the issue and its potential impact.
* Steps to reproduce, including the configuration used (especially `server.host`,
  `server.allow_lan_exposure`, and any provider/backend configuration).
* The component version (`modelrack --version` for applications, or the installed
  package version for libraries).

You should expect an acknowledgement within a reasonable time and, once a fix is available, credit in
the release notes unless you ask otherwise.

## Scope

In scope: this repository's own code and its documented configuration surface. Vulnerabilities in a
model provider (e.g. Ollama itself), in the operating system, or in a third-party dependency should
be reported to that project directly — `pip-audit` runs in this repository's CI to catch known
vulnerable dependency versions.

## Security-relevant design decisions

The rationale for this component's security posture is recorded, not assumed. Start with:

* `docs/standards/security-standards.md` — trust boundaries, network exposure, input validation,
  filesystem safety, secrets handling.
* `docs/adr/0014-authentication-strategy.md` — bearer tokens, scopes, loopback-vs-LAN behaviour.
* `docs/adr/0026-local-http-hardening.md` — Host header validation, CSRF, outbound-fetch allowlisting.
