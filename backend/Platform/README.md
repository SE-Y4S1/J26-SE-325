# Platform Service

Identity and portfolio storage shared by all four components of **J26-SE-325**.

Deliberately owns nothing domain-specific. Component 1 forecasts and optimizes; Components
2–4 detect fraud, anchor audits and explain. All of them need to know who the user is and
what they hold, and none should have to depend on another team's service to find out.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/auth/register` | create an account, returns a token |
| POST | `/auth/login` | exchange credentials for a token |
| GET | `/auth/me` | current user |
| GET / POST | `/portfolios` | list / create |
| GET / PUT / DELETE | `/portfolios/{id}` | read / update / delete |
| GET | `/health` | liveness + database check |

## Running

```bash
uv venv --python 3.12
uv pip install fastapi "uvicorn[standard]" pydantic pydantic-settings email-validator \
               sqlalchemy pyjwt "passlib[argon2]" argon2-cffi python-dotenv pytest httpx

JWT_SECRET=$(python -c "import secrets; print(secrets.token_urlsafe(48))") \
  uv run uvicorn service.api:app --port 8100 --reload
```

`JWT_SECRET` is shared with every component: this service **issues** tokens, each component
**verifies** them, and no service calls another on the request path. It refuses to start if
the secret is unset, under 32 characters, or still the placeholder from `.env.example`.

## Design notes

**Holdings mirror Component 1's contract exactly** — `symbol`, `quantity`, `current_price`,
`avg_daily_volume`, `cost_basis`. The frontend posts a stored portfolio's holdings straight
to `/portfolio/withdraw`; a translation layer between two contracts meant to be identical is
where field drift hides.

**Argon2, not bcrypt.** Argon2id is the current OWASP recommendation and has no length
ceiling — bcrypt silently truncates at 72 bytes, quietly weakening a long passphrase with no
error anywhere.

**Cross-user access returns 404, not 403.** A 403 confirms the id is real, which lets anyone
enumerate how many portfolios the platform holds.

**Login is not an enumeration oracle.** Unknown email and wrong password return an identical
status and body.

**`holdings=None` and `holdings=[]` mean different things** on update: leave them alone
versus remove them all. Without that distinction, renaming a portfolio would force the client
to resend every holding.

**SQLite via SQLAlchemy.** The TAF scopes balances as *"simulated platform balances rather
than real brokerage holdings"*, so there is no durability requirement justifying Postgres.
The ORM earns its place here even though Component 1's model registry uses raw `sqlite3`:
that registry is one flat table, whereas users own portfolios which own holdings, and
hand-writing those cascades is where bugs live. In-memory URLs use a `StaticPool` so the
whole app shares one connection — without it, `create_all` and the request handlers each get
their own empty database.

## Tests

```bash
uv run pytest -q          # 40 tests
```

The security-relevant ones assert what the API does *not* reveal: identical responses for
unknown-email and wrong-password, 404 for another user's data, rejection of `alg: none` and
foreign-secret tokens, and that no response ever contains a password or its hash.
