# Domain Docs

This repository uses a single-context domain-documentation layout.

## Before exploring

Read:

1. `CONTEXT.md` at the repository root.
2. ADRs under `docs/adr/` that touch the area being changed.

If either location does not exist, proceed silently. Domain-modeling workflows create or extend these files when terminology or architectural decisions are resolved.

## Layout

```text
/
├── CONTEXT.md
├── docs/
│   └── adr/
└── crates/
```

`CONTEXT.md` is the shared glossary for the entire Cargo workspace. Keep it free of implementation details, plans, and architectural rationale.

`docs/adr/` records system-wide decisions. All workspace crates belong to the same technical-analysis context.

## Vocabulary

Use the terms defined in `CONTEXT.md` in issue titles, specifications, tests, and implementation proposals. Do not drift to synonyms that the glossary explicitly avoids.

If a required concept is missing, first determine whether the proposed language is unnecessary. If it represents a real domain gap, resolve it through the domain-modeling workflow.

## ADR conflicts

Explicitly identify proposals that contradict an existing ADR rather than silently overriding it:

> Contradicts ADR-0007 — worth reconsidering because…
