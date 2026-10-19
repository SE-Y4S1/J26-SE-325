# Deploying Nexa (Render free plan + Prisma Postgres + Vercel)

```
 Browser ── Vercel (frontend) ──┬─> nexa-platform  (Render) ──┬─> nexa-fraud (Render)
                                ├─> nexa-portfolio (Render)   └─> nexa-audit (Render)
                                ├─> nexa-fraud
                                └─> nexa-audit
 nexa-platform and nexa-audit ──> Prisma Postgres (one database, separate tables)
```

Component 4 (assistant) is not deployed: it needs a local Ollama model (~4 GB RAM), far
beyond the free plan's 512 MB. The Assistant screen shows its "service not running" state.

## 1. Prisma Postgres

1. Create a database at [console.prisma.io](https://console.prisma.io).
2. Copy the **direct connection string**, the one that starts `postgres://` and points at
   `db.prisma.io:5432`. Not the `prisma+postgres://accelerate...` URL: that one speaks
   Prisma's own protocol and neither service can use it (the platform refuses it at startup
   with this same instruction).

Both services create their own tables on first start (`users`, `portfolios`, `holdings`,
`executions`; `audit_records`, `audit_mock_ledger`). No migration step.

## 2. Render

1. Render dashboard → **New → Blueprint** → this repository, branch `main`. It reads
   [`render.yaml`](render.yaml) and proposes four free web services.
2. Fill in the values it asks for:

| Variable | Services | Value |
|---|---|---|
| `DATABASE_URL` | nexa-platform, nexa-audit | the direct Prisma connection string (same one for both) |
| `ALLOWED_ORIGINS` | platform, portfolio, fraud | your Vercel URL, e.g. `https://your-app.vercel.app` (comma-separate several) |
| `FRAUD_SERVICE_URL` | nexa-platform | public URL of nexa-fraud, e.g. `https://nexa-fraud.onrender.com` |
| `AUDIT_SERVICE_URL` | nexa-platform | public URL of nexa-audit |

   Render shows each service's URL on its page once created. If a name was taken it adds a
   suffix, so copy the real URLs rather than assuming them, then update the two `*_SERVICE_URL`
   values and redeploy nexa-platform.
3. `JWT_SECRET` is generated once in the `nexa-shared` group and shared by platform and
   portfolio automatically.

The portfolio and fraud images install CPU PyTorch, so their first builds take a while.

## 3. Vercel

Project → Settings → Environment Variables (Production), then **redeploy**: `NEXT_PUBLIC_*`
values are baked in at build time, so changing them without a rebuild does nothing.

| Variable | Value |
|---|---|
| `NEXT_PUBLIC_PLATFORM_URL` | `https://nexa-platform.onrender.com` |
| `NEXT_PUBLIC_PORTFOLIO_URL` | `https://nexa-portfolio.onrender.com` |
| `NEXT_PUBLIC_FRAUD_URL` | `https://nexa-fraud.onrender.com` |
| `NEXT_PUBLIC_AUDIT_URL` | `https://nexa-audit.onrender.com` |
| `NEXT_PUBLIC_ASSISTANT_URL` | leave unset (Component 4 is not hosted) |

Use the real Render URLs from step 2.

## What to expect on the free plan

* **Cold starts.** A service idle for 15 minutes sleeps; the next request waits about a
  minute while it wakes (longer for the PyTorch services). The platform waits up to 120 s for
  the fraud and audit services; if one is still waking the execution is **HELD**, never run
  unscreened, and **Retry** completes it.
* **750 free instance-hours a month** are shared by all four services. Sleeping services do
  not consume them.
* **No forecasting models or FinBERT.** Their weights are not in the image, so `/forecast`
  answers 503 and news sentiment is unavailable; optimize and withdraw work. The market-data
  cache and MLflow runs live on the ephemeral disk and rebuild on demand.
* **Fraud scores come from the untrained heuristic** unless a trained bundle is baked into
  the image; every response says which (`model_version`, `benchmark_model`).
* **The audit ledger is simulated**, persisted in Postgres so commitments survive restarts.
  For real anchoring, deploy `AuditRegistry.sol` to a public testnet (e.g. Sepolia, using a
  free RPC endpoint and faucet funds) and set `BLOCKCHAIN_MODE` unset, `BLOCKCHAIN_RPC_URL`,
  `SIGNER_PRIVATE_KEY` (as a secret) and `CONTRACT_CONFIG_PATH` on nexa-audit.
* **Data**: accounts, portfolios, executions and audit records persist in Prisma Postgres.
