# Model-provider route validation

AutoBench must verify the supervision model route before any hardware workload
is scheduled. A configured route is not proof that the requested model was
actually used.

## Canonical Luna route

The required route for Luna supervision is:

```text
litellm-edge/cl/gpt-5.6-luna
```

The direct-provider route below is not equivalent and must not be used for this
project:

```text
openai/gpt-5.6-luna
```

The direct route may be rejected by the OpenAI endpoint even when local auth
metadata says that an OpenAI credential exists. A resolver/completion check is
required; credential presence alone is insufficient.

## Required gate

Before `run_remote.py` deploys or schedules a workload when a required route is
specified, the supervisor must provide sanitized evidence for:

- configured provider-qualified route;
- resolved provider;
- resolved model;
- identity-check result (`verified`, `auth_failed`, `rejected`, or `unverified`);
- validation timestamp.

The gate accepts only an exact provider/model identity match and
`identity_check=verified`. It blocks invalid routes, authentication rejection,
identity mismatch, missing evidence, and ambiguous identity. In particular, a
failed Luna request must not silently continue on a Gemini fallback.

Example validation flags:

```text
python scripts/run_remote.py \
  --required-model-route litellm-edge/cl/gpt-5.6-luna \
  --configured-model-route litellm-edge/cl/gpt-5.6-luna \
  --resolved-provider litellm-edge \
  --resolved-model cl/gpt-5.6-luna \
  --identity-check verified \
  --deploy-only
```

These flags carry identity metadata only. Never pass credentials, authorization
headers, raw provider payloads, prompts, responses, private endpoints, or host
identifiers.

## Failure policy

A route-validation failure is a supervision/configuration blocker, not a GPU
failure and not evidence of model quality. Stop before workload scheduling and
report one of the sanitized statuses:

- `MODEL_ROUTE_INVALID`;
- `MODEL_ROUTE_AUTH_FAILED`;
- `MODEL_ROUTE_IDENTITY_MISMATCH`;
- `MODEL_ROUTE_IDENTITY_UNVERIFIED`.

Focused tests cover canonical-route acceptance, incorrect direct-provider route,
authentication rejection/fallback suppression, and missing or ambiguous identity
without invoking hardware inference.
