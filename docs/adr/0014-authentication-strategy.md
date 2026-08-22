# ADR-0014 — Authentication strategy

**Status:** Accepted (2026-08-21)

## Context

The default deployment is a single user on their own machine, where any authentication requirement
is pure friction. The supported LAN deployment (LoadCoach on a GPU host, IdeaPress on a laptop) puts
an inference service — capable of consuming the whole GPU and reading the user's projects — on a
network. Those two facts pull in opposite directions.

Requirements: bind to localhost by default; require explicit configuration for LAN exposure;
authenticate for non-local exposure; hash stored API tokens; least privilege.

## Decision

**No authentication on loopback. Mandatory bearer tokens for any non-loopback binding.**

1. Default: bind `127.0.0.1`, no tokens configured, no authentication. The OS user boundary is the
   security boundary.
2. Binding to any other address requires **both** an explicit `server.host` and at least one
   configured API token. Missing either ⇒ startup refuses with `INSECURE_BINDING` (exit 3).
   `0.0.0.0` additionally requires `server.allow_lan_exposure = true`, so a typo cannot expose a
   service.
3. Scheme: `Authorization: Bearer <token>`. Tokens are 32 random bytes from `secrets.token_bytes`,
   base32-encoded, prefixed with the application (`lc_…`) for identification in logs and secret
   scanners.
4. **Storage: SHA-256 hash only.** The plaintext is shown once at creation and never stored, never
   logged. Comparison uses `hmac.compare_digest`.
5. Scopes: `read`, `write`, `admin`, checked in the service layer as well as at the route.
6. Failed authentication is rate-limited per source address and logged with source and request ID,
   never with the presented token.
7. TLS is terminated by a reverse proxy; the applications speak HTTP and warn at startup when bound
   non-loopback without evidence of a proxy.
8. Tokens are managed by CLI: `<app> token create|list|revoke`.

## Alternatives considered

**Always require authentication, even on loopback.** Rejected: it breaks zero-configuration startup
and adds a token to every `curl` on a single-user machine, buying nothing against an attacker who
already has local code execution.

**Session cookies + a login page.** Rejected: it implies user accounts, password storage and
password-reset flows — a whole subsystem for a product whose deployment model is one user, or a few
users behind a proxy that already knows who they are. Bearer tokens serve both the UI (stored in
`localStorage` when auth is on) and scripts.

**OAuth2 / OIDC.** Rejected: an identity provider is not something a local-first tool may require.
Recorded as a future extension for a genuine multi-user deployment.

**mTLS.** Rejected: certificate management burden far exceeds the threat for a home or small-team LAN.

**Argon2id or bcrypt for token hashing.** Rejected with reasoning, because it is the non-obvious
call: slow KDFs defend *low-entropy* secrets against offline brute force. These tokens are 256 bits
of CSPRNG output; SHA-256 of such a value is not brute-forceable, and a fast hash keeps
per-request verification cheap. A password-based scheme would require a KDF — which is one reason
passwords are not used.

**Trust `X-Forwarded-For` / proxy-supplied identity.** Rejected as a default: it is only safe with a
correctly configured proxy. Available as explicit configuration for deployments that have one.

## Consequences

*Positive.* Zero friction locally; a real barrier when exposed. Refusing to start on an unsafe
combination makes accidental exposure structurally difficult. Tokens work identically for the UI,
the CLI and third-party clients. Hashed storage means a leaked database yields no usable credential.

*Negative.* No user identity — audit logs attribute actions to a token name, not a person. Acceptable
for the target deployments and documented plainly.

*Negative.* No token rotation policy beyond manual revoke/create. Optional expiry is supported;
mandatory rotation is not.

*Negative.* Anyone with local shell access can read the config and the database. Documented honestly:
the OS user boundary is the boundary, and full-disk encryption is the user's responsibility.

## Revisit when

A genuine multi-user deployment with per-user attribution and access control appears. The migration
path is: keep bearer tokens as the machine credential, add an identity layer (OIDC) for humans, and
map both onto the existing scope checks — which is why scopes are enforced in the service layer
rather than only at the route.
