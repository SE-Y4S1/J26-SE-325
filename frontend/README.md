# Frontend

Web app for the **J26-SE-325** platform. Next.js 16.3 · React 19 · Tailwind v4.

## Running

```bash
cp .env.example .env.local
npm install
npm run dev
```

The backends must be up too — see the repository-root
[`INTEGRATION.md`](../INTEGRATION.md), or run everything with `docker compose up --build`
from the root.

| Script | Purpose |
|---|---|
| `npm run dev` | dev server on :3000 |
| `npm run build` | production build — **also typechecks**, which `next dev` does not |
| `npm run gen:api` | regenerate TypeScript types from each service's live OpenAPI |
| `npm run typecheck` | `tsc --noEmit` |
| `npx eslint .` | lint (`next lint` was removed in Next 16) |

## Structure

```
app/
  (auth)/login  register       unauthenticated pages
  (platform)/                  shared shell + auth guard
    portfolio/                 holdings CRUD
    withdraw/                  Component 1's headline screen
    optimize/                  MOEA/D allocation
    forecast/                  quantile forecasts
    fraud/ audit/ assistant/   placeholders for Components 2-4
lib/
  api/client.ts                fetch wrapper: base URLs, bearer token, typed errors
  api/platform.ts portfolio.ts typed clients, one per service
  api/generated/               GENERATED from OpenAPI — never hand-edit
  auth/context.tsx             AuthProvider + useAuth
components/                    shared primitives
```

A component owns one route folder and one client module. Adding Component 2 means adding
`app/(platform)/fraud/page.tsx` and `lib/api/fraud.ts`, then flipping `ready: true` in `NAV`
in `app/(platform)/layout.tsx`. Nothing else changes.

## Types are generated, not written

`lib/api/generated/*.ts` comes from each service's `/openapi.json` via `npm run gen:api`.
Component 1's `service/contracts.py` is a **frozen** contract that Components 3 and 4 also
bind to; a hand-copied interface would drift from it silently, whereas a generated file turns
any backend change into a reviewable diff. `lib/api/types.ts` aliases the generated types so
component code stays readable without anyone redeclaring a shape.

The one hand-written type is `FuzzyRuleTraceEntry`. The backend declares that field as
`list[dict]`, so OpenAPI cannot describe it; it is produced by
`optimization/fuzzy_withdrawal.py::rule_trace_to_dict`, and if that changes this must too.

## Next.js 16 — what differs from older guides

Verified against `node_modules/next/dist/docs/`, per `AGENTS.md`:

- **`fetch` is not cached by default**, but routes still prerender at build, so a build would
  bake a response captured at build time. Every API call passes `cache: "no-store"`.
- **Async request APIs are mandatory** — `params`, `searchParams`, `cookies()`, `headers()`
  must be awaited. Synchronous access was removed, not deprecated.
- **`middleware.ts` is now `proxy.ts`**, Node runtime only.
- **`next lint` was removed**; `next build` no longer lints.
- **Turbopack is the default**; a `webpack` key in `next.config.ts` fails the build.
- Generated `LayoutProps<"/">` / `PageProps<...>` types are global — `app/layout.tsx` uses
  that style and new routes should match.
- **Tailwind v4 is CSS-first.** There is no `tailwind.config.ts`; theme tokens live in
  `@theme inline` in `app/globals.css`.
- `@/*` resolves to `frontend/`, not `frontend/src/`.

Anything reading `useSearchParams` must sit inside a `<Suspense>` boundary, or the whole route
is forced to client-side rendering at build.

## Behaviours that look like bugs and are not

| Observation | Explanation |
|---|---|
| Forecast screen shows "No trained forecaster is registered" | Expected. `/forecast` returns 503 until Colab fine-tuning has run; every other screen works without it. |
| A withdrawal reports `feasible: false` | A real answer, not a failure — the daily participation cap limits how much can be liquidated in the deadline. The shortfall is shown deliberately. |
| `model_version` reads `unregistered` | No checkpoint registered yet. The withdrawal path does not need one. |
| Optimize takes ~7s | MOEA/D runs 100 generations over 45 reference directions server-side. |

## Security note

The access token is held in browser storage and sent to each backend directly, so an XSS
would expose it. A backend-for-frontend proxy would have allowed an httpOnly cookie the page
cannot read. This is a documented trade-off of the direct-CORS architecture — see the
security section of [`INTEGRATION.md`](../INTEGRATION.md).
