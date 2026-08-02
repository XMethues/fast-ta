# Issue tracker: GitHub

Issues and PRDs for this repository live in GitHub Issues under `XMethues/fast-ta`. Use the `gh` CLI for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, including labels and relevant comments.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments` with appropriate label and state filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`.
- **Apply or remove labels**: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- **Close an issue**: `gh issue close <number> --comment "..."`.

Infer the repository from `git remote -v`; `gh` does this automatically when run inside the clone.

## Pull requests as a triage surface

**PRs as a request surface: no.** _(Set to `yes` if this repository later treats external PRs as feature requests.)_

When set to `yes`, PRs use the same labels and states as issues through the corresponding `gh pr` commands. GitHub shares one number space across issues and PRs, so resolve an ambiguous `#42` with `gh pr view 42` and fall back to `gh issue view 42`.

## Skill operations

- When a skill says **publish to the issue tracker**, create a GitHub issue.
- When a skill says **fetch the relevant ticket**, run `gh issue view <number> --comments`.
- Fetch labels and comments whenever a skill needs the complete ticket state.

## Wayfinding operations

Used by `/wayfinder`. A map is one issue with child issues as tickets.

- **Map**: an issue labelled `wayfinder:map`, holding Notes, Decisions-so-far, and Fog.
- **Child ticket**: a GitHub sub-issue where supported. Otherwise, link it from a task list in the map and put `Part of #<map>` at the top of the child. Use `wayfinder:<type>` labels: `research`, `prototype`, `grilling`, or `task`.
- **Blocking**: use GitHub native issue dependencies. If unavailable, add `Blocked by: #<number>` at the top of the child.
- **Frontier**: the first open, unassigned child in map order with no open blocker.
- **Claim**: `gh issue edit <number> --add-assignee @me`.
- **Resolve**: comment with the result, close the child, and append a context pointer to the map's Decisions-so-far.
