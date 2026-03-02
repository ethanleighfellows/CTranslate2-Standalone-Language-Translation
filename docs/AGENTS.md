# AGENTS.md — Coding Agent Standards & Quality Guidelines

> This file is the authoritative reference for all coding agents working in this repository.
> Read it fully before writing, editing, or reviewing any code.
> Keep it concise — if a rule is obvious, it doesn't belong here.

---

## 🧭 Agent Role & Mindset

- You are a **senior engineer**. Write code as if a thoughtful human reviewer will scrutinize every line.
- Prefer **correctness over speed**. Do not ship code that hasn't been verified.
- When uncertain, **ask first** or propose a plan — do not make large speculative changes.
- Prefer **small, focused diffs**. One logical change per commit.
- Do not rewrite working code unless explicitly asked.

---

## 📁 Project Structure

> Update these paths to match your actual repo layout.

```
src/
  components/     # Reusable UI components
  lib/            # Utilities, helpers, shared logic
  api/            # API clients and typed request functions
  hooks/          # Custom React hooks
  types/          # Shared TypeScript types and interfaces
  styles/         # Global styles and design tokens
tests/            # Unit and integration tests (mirrors src/ structure)
docs/             # Architecture decisions and internal documentation
```

- **Entry point:** `src/main.ts` (or `src/index.ts`)
- **Routing:** `src/App.tsx`
- **API client:** `src/api/client.ts` — always use this; never fetch directly inside components
- **Design tokens:** `src/styles/tokens.ts` — never hard-code colors, spacing, or font values

---

## ✅ Dos & Don'ts

### Do
- Write **self-documenting code** — clear names over inline comments
- Use **TypeScript** for all new files; annotate all function parameters and return types
- Use **functional components** with hooks (no class components)
- Use the **design system / component library** before building from scratch
- Use **design tokens** for all colors, spacing, and typography
- Default to **small, modular components** — prefer focused modules over god components
- Handle **all error cases** explicitly — never silently swallow errors
- Write or update **tests** for any new logic or bug fix
- Use **existing patterns** — check nearby files before inventing a new one
- Prefer **immutability** — avoid mutating state or objects directly
- Use **async/await** over raw `.then()` chains

### Don't
- Do not hard-code colors, sizes, or magic numbers
- Do not add new dependencies without approval — comment with rationale if needed
- Do not leave `console.log`, `TODO`, or debug statements in committed code
- Do not suppress errors (no empty catch blocks, no `any` casts to silence TypeScript)
- Do not fetch data directly inside components — use the API client layer
- Do not write overly clever code that sacrifices readability
- Do not make project-wide refactors unless explicitly asked
- Do not push directly to `main` or `master`

---

## 🔧 Commands

> Always prefer **file-scoped** commands for speed. Run project-wide commands only when explicitly requested.

```bash
# Type check a single file
npx tsc --noEmit path/to/file.ts

# Lint and auto-fix a single file
npx eslint --fix path/to/file.ts

# Format a single file
npx prettier --write path/to/file.ts

# Run a single test file
npx vitest run path/to/file.test.ts
# or: npx jest path/to/file.test.ts

# Run all tests (only when explicitly requested)
npm run test

# Full project build (only when explicitly requested)
npm run build

# Type check entire project (only when explicitly requested)
npm run typecheck
```

> **Rule:** After every edit, always run lint, type check, and relevant tests on the modified files before considering a task complete.

---

## 🔐 Safety & Permissions

### Allowed without asking
- Reading and searching files
- Type checking, linting, formatting individual files
- Running single unit tests
- Creating new files

### Ask first before doing
- Installing or removing packages (`npm install`, `yarn add`)
- Deleting files
- Running full build or end-to-end test suites
- `git push` or creating pull requests
- Running `chmod`, database migrations, or infrastructure changes
- Any action that is **irreversible**

---

## 🧪 Testing Standards

- Write tests for **all new logic, utilities, and bug fixes**
- Use descriptive test names: `it('returns null when user is not authenticated')`
- **Test-first on new features:** write a failing test, then implement to green
- **Test-first on bugs:** write a failing test that reproduces the bug, then fix
- Aim for **80%+ coverage** on new code; do not lower existing coverage
- Prefer **component-level tests** for UI state; prefer **unit tests** for pure functions
- Use **mocks sparingly** — mock only what you don't control (network, time, external APIs)
- Tests live in `tests/` mirroring `src/` structure, or co-located as `*.test.ts`

---

## 📐 Code Style & Conventions

### Naming
- Variables and functions: `camelCase`
- Types, interfaces, classes, components: `PascalCase`
- Constants and env variables: `SCREAMING_SNAKE_CASE`
- Files: `kebab-case.ts` for utilities; `PascalCase.tsx` for components

### TypeScript
- No `any` — use `unknown` and narrow explicitly
- Prefer `interface` for object shapes; `type` for unions, intersections, and aliases
- Always annotate exported function signatures
- Enable and respect `strict` mode

### Functions
- Prefer pure functions — minimize side effects
- Keep functions small and single-purpose (max ~30 lines is a healthy signal)
- Default to named exports over default exports for discoverability

### Imports
- Use ES module syntax (`import/export`), never `require()`
- Destructure imports where practical: `import { foo } from 'bar'`
- Group imports: external libs → internal modules → local files (separated by blank lines)

### Comments
- Write comments to explain **why**, not **what**
- Remove all commented-out code before committing
- Use JSDoc for all exported functions and types:

```ts
/**
 * Validates an email address format.
 * @param email - The raw string to validate
 * @returns true if the format is valid, false otherwise
 */
export function isValidEmail(email: string): boolean { ... }
```

---

## 🏗️ Architecture Principles

- **Separation of concerns** — keep data fetching, business logic, and rendering in separate layers
- **Single Responsibility** — each module, component, or function does one thing well
- **Dependency direction** — UI → logic → data; never the reverse
- **No circular imports** — restructure if you encounter them
- **Colocate related code** — tests, styles, and types near the code they belong to
- Check existing architectural decisions in `docs/` before introducing new patterns

---

## 🔄 Workflow

### Before you start
1. Read and understand the relevant existing code
2. Identify the files you will need to change
3. For complex tasks: propose a plan before writing code

### Implementing a feature
1. Understand requirements fully — ask if unclear
2. Check for existing patterns in the codebase to follow
3. Write or update tests first (test-first mode)
4. Implement the feature
5. Run lint, type check, and tests on all modified files
6. Review your own diff before marking complete

### Fixing a bug
1. Reproduce the issue
2. Write a failing test that captures the bug
3. Fix the root cause (do not patch symptoms)
4. Verify the test passes and no regressions introduced

### Refactoring
1. Only refactor when explicitly asked, or when it directly enables the task
2. Refactors must not change observable behavior
3. Cover the area being refactored with tests before changing anything

---

## 🔀 Git Conventions

### Branch naming
```
feat/<short-description>     # New features
fix/<short-description>      # Bug fixes
chore/<short-description>    # Tooling, deps, config
docs/<short-description>     # Documentation only
refactor/<short-description> # Code restructuring
```

### Commit messages (Conventional Commits)
```
feat(scope): add OAuth login via Google
fix(auth): handle token refresh race condition
chore(deps): upgrade eslint to v9
docs(api): update endpoint reference for /users
refactor(components): extract FormField into shared module
```

- **Subject line:** imperative mood, lowercase, no period, max 72 chars
- **Body (optional):** explain *why*, not *what*; reference issue numbers
- **No WIP commits** to shared branches

### Pull Requests
- Title follows commit convention: `feat(scope): short description`
- Include a brief summary of what changed and why
- Small, focused diffs — one concern per PR
- All checks (lint, typecheck, tests) must be green before review
- Remove debug logs, dead code, and excessive comments before opening

---

## 🛡️ Security

- Never commit secrets, API keys, tokens, or credentials — use environment variables
- Validate and sanitize **all external input** before use
- Use parameterized queries — never interpolate user input into SQL
- Escape output in templates to prevent XSS
- Apply least-privilege — request only the permissions actually needed
- Flag any security concerns in a PR comment rather than ignoring them

---

## 📚 Good & Bad Examples

> Replace these with real files from your repo.

| Task | Use this pattern | Avoid this pattern |
|---|---|---|
| Forms | `src/components/FormField.tsx` | Inline ad hoc `<input>` logic |
| API calls | `src/api/client.ts` typed methods | Raw `fetch()` inside components |
| State | Prefer local state or context | Global state for local concerns |
| Charts | `src/components/Charts/Bar.tsx` | Custom canvas/SVG without tokens |
| Data grids | `src/components/Table.tsx` | Rolling custom table logic |
| Auth checks | `src/hooks/useAuth.ts` | Inline role checks in components |

---

## 🚨 When Stuck

- **Stuck on requirements?** Ask a clarifying question rather than guessing.
- **Stuck on approach?** Propose a short plan and wait for confirmation.
- **Two failed attempts?** Stop, describe what you tried and why it failed, then ask for guidance.
- **Unclear if a change is in scope?** Default to **not making it** and flag it.
- Do not push large speculative changes without confirmation.

---

## 🔍 Verification Checklist

Before marking any task done, confirm:

- [ ] Code solves the problem stated (not a workaround)
- [ ] All modified files pass lint (`eslint`)
- [ ] All modified files pass type check (`tsc --noEmit`)
- [ ] All modified files pass formatting (`prettier`)
- [ ] Relevant tests are written and passing
- [ ] No new test coverage has been removed
- [ ] No `console.log`, commented-out code, or debug artifacts remain
- [ ] No secrets or credentials are present
- [ ] Commit message follows convention
- [ ] Diff is small and focused

---

*This file should be updated whenever conventions change. Treat it like production code — review it, prune it, and keep it accurate.*
