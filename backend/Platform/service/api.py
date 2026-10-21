"""Platform service: identity and portfolio storage shared by all four components.

Deliberately owns nothing domain-specific. Component 1 forecasts and optimizes; Components
2-4 detect fraud, anchor audits and explain. All of them need to know who the user is and
what they hold, and none of them should have to depend on another component's service to
find out.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth.deps import get_current_user
from auth.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    get_jwt_secret,
    hash_password,
    verify_password,
)
from service import pipeline
from service.contracts import (
    ExecutionConfirm,
    ExecutionRequest,
    ExecutionResponse,
    ExecutionSummary,
    HealthResponse,
    HoldingModel,
    LoginRequest,
    PortfolioCreate,
    PortfolioResponse,
    PortfolioSummary,
    PortfolioUpdate,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from service.cors import install_cors
from store.database import get_session, init_db
from store.models import Execution, Holding, Portfolio, User

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Validate the secret at STARTUP, not on the first login. A service that boots happily
    # and only fails when a user tries to sign in is much harder to diagnose.
    get_jwt_secret()
    init_db()
    logger.info("platform service ready")
    yield


app = FastAPI(
    title="J26-SE-325 Platform Service",
    version="0.1.0",
    description="Identity and portfolio storage shared by all four platform components.",
    lifespan=lifespan,
)
install_cors(app)


# --------------------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health(session: Session = Depends(get_session)) -> HealthResponse:
    try:
        session.execute(select(User).limit(1))
        database_ready = True
    except Exception as exc:  # noqa: BLE001
        logger.error("database health check failed: %s", exc)
        database_ready = False

    return HealthResponse(status="ok", service="platform", database_ready=database_ready)


# --------------------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------------------

@app.post("/auth/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, session: Session = Depends(get_session)) -> TokenResponse:
    """Create an account and return a token, so the client is not forced to log in again."""
    normalised = request.email.lower().strip()

    user = User(
        email=normalised,
        display_name=request.display_name.strip(),
        password_hash=hash_password(request.password),
    )
    session.add(user)
    try:
        session.commit()
    except IntegrityError:
        # Rely on the unique constraint rather than a pre-check SELECT: the check-then-insert
        # pattern races two concurrent registrations of the same address.
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with that email already exists",
        ) from None

    session.refresh(user)
    logger.info("registered user %s", user.id)
    return _token_for(user)


@app.post("/auth/login", response_model=TokenResponse)
def login(request: LoginRequest, session: Session = Depends(get_session)) -> TokenResponse:
    """Exchange credentials for an access token."""
    user = session.scalar(select(User).where(User.email == request.email.lower().strip()))

    # Same 401 whether the address is unknown or the password is wrong. Distinguishing them
    # turns the login form into an account-enumeration oracle.
    if user is None or not verify_password(request.password, user.password_hash):
        logger.info("failed login for %s", request.email)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _token_for(user)


@app.get("/auth/me", response_model=UserResponse)
def me(user: User = Depends(get_current_user)) -> UserResponse:
    return UserResponse.model_validate(user)


def _token_for(user: User) -> TokenResponse:
    return TokenResponse(
        access_token=create_access_token(subject=str(user.id), extra_claims={"email": user.email}),
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserResponse.model_validate(user),
    )


# --------------------------------------------------------------------------------------
# Portfolios
# --------------------------------------------------------------------------------------

@app.get("/portfolios", response_model=list[PortfolioSummary])
def list_portfolios(
    user: User = Depends(get_current_user), session: Session = Depends(get_session)
) -> list[PortfolioSummary]:
    portfolios = session.scalars(
        select(Portfolio).where(Portfolio.user_id == user.id).order_by(Portfolio.updated_at.desc())
    ).all()

    return [
        PortfolioSummary(
            id=p.id, name=p.name, base_currency=p.base_currency,
            holding_count=len(p.holdings), total_value=p.total_value, updated_at=p.updated_at,
        )
        for p in portfolios
    ]


@app.post("/portfolios", response_model=PortfolioResponse, status_code=status.HTTP_201_CREATED)
def create_portfolio(
    request: PortfolioCreate,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> PortfolioResponse:
    portfolio = Portfolio(
        user_id=user.id, name=request.name.strip(), base_currency=request.base_currency.upper(),
        holdings=[_to_holding(h) for h in request.holdings],
    )
    session.add(portfolio)
    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"You already have a portfolio named {request.name!r}",
        ) from None

    session.refresh(portfolio)
    return _to_response(portfolio)


@app.get("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
def get_portfolio(
    portfolio_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> PortfolioResponse:
    return _to_response(_owned_portfolio(portfolio_id, user, session))


@app.put("/portfolios/{portfolio_id}", response_model=PortfolioResponse)
def update_portfolio(
    portfolio_id: int,
    request: PortfolioUpdate,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> PortfolioResponse:
    """Partial update. `holdings=None` leaves them untouched; `holdings=[]` clears them."""
    portfolio = _owned_portfolio(portfolio_id, user, session)

    if request.name is not None:
        portfolio.name = request.name.strip()
    if request.base_currency is not None:
        portfolio.base_currency = request.base_currency.upper()
    if request.holdings is not None:
        # Replace wholesale. cascade="all, delete-orphan" removes the detached rows, so this
        # does not leak orphaned holdings the way a plain reassignment would.
        portfolio.holdings.clear()
        session.flush()
        portfolio.holdings.extend(_to_holding(h) for h in request.holdings)

    try:
        session.commit()
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="That name is already used by another of your portfolios",
        ) from None

    session.refresh(portfolio)
    return _to_response(portfolio)


@app.delete("/portfolios/{portfolio_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_portfolio(
    portfolio_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> None:
    session.delete(_owned_portfolio(portfolio_id, user, session))
    session.commit()


# --------------------------------------------------------------------------------------
# Executions (see service/pipeline.py for the flow and its failure policy)
# --------------------------------------------------------------------------------------

@app.post(
    "/portfolios/{portfolio_id}/executions",
    response_model=ExecutionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_execution(
    portfolio_id: int,
    request: ExecutionRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> ExecutionResponse:
    """Screen a Component 1 plan with Component 2, anchor it with Component 3, then execute.

    201 whatever the outcome: the execution record exists either way, and `status` says
    whether holdings changed. Only a plan that cannot apply to this portfolio is a 422.
    """
    portfolio = _owned_portfolio(portfolio_id, user, session)
    held = {h.symbol: {"quantity": h.quantity, "current_price": h.current_price}
            for h in portfolio.holdings}
    try:
        orders = pipeline.derive_orders(request.kind, request.plan, held)
    except pipeline.PlanError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None

    execution = Execution(
        id=pipeline.new_execution_id(),
        user_id=user.id,
        portfolio_id=portfolio.id,
        kind=request.kind,
        status="HELD",
        amount=pipeline.transaction_amount(request.kind, request.plan, orders),
        currency=_plan_currency(request.plan, portfolio.base_currency),
        source_model_version=str(request.plan.get("model_version") or "unknown"),
        plan_hash=pipeline.plan_hash(request.kind, portfolio.id, request.plan, orders),
        plan=request.plan,
        orders=orders,
    )
    _run_stages(execution, portfolio, user.id, request.device_key)
    session.add(execution)
    session.commit()
    session.refresh(execution)
    return ExecutionResponse.model_validate(execution)


@app.get("/executions", response_model=list[ExecutionSummary])
def list_executions(
    portfolio_id: int | None = None,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[ExecutionSummary]:
    query = select(Execution).where(Execution.user_id == user.id)
    if portfolio_id is not None:
        query = query.where(Execution.portfolio_id == portfolio_id)
    rows = session.scalars(query.order_by(Execution.created_at.desc()).limit(100)).all()
    return [ExecutionSummary.model_validate(r) for r in rows]


@app.get("/executions/{execution_id}", response_model=ExecutionResponse)
def get_execution(
    execution_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> ExecutionResponse:
    """The full trace: Component 1's plan, Component 2's decision, Component 3's anchor."""
    return ExecutionResponse.model_validate(_owned_execution(execution_id, user, session))


@app.post("/executions/{execution_id}/confirm", response_model=ExecutionResponse)
def confirm_execution(
    execution_id: str,
    request: ExecutionConfirm,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> ExecutionResponse:
    """Step-up for a PENDING_REVIEW execution. The review is anchored before anything moves."""
    execution = _owned_execution(execution_id, user, session)
    if execution.status != "PENDING_REVIEW":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"Execution is {execution.status}, not awaiting review")
    # 400, not 401: the session is valid, and a 401 would sign the user out of the app.
    if not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Password did not match; the execution is unchanged")

    review = pipeline.submit_review(execution.id, request.approve)
    if not review.reachable or review.status_code != 200:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The audit bridge (Component 3) could not record the review; the execution is unchanged",
        )
    execution.audit = {**(execution.audit or {}), "review": pipeline.audit_record(review)}

    if not request.approve:
        execution.status, execution.status_reason = "REJECTED", "Rejected by the account holder at step-up."
    else:
        portfolio = session.get(Portfolio, execution.portfolio_id) if execution.portfolio_id else None
        if portfolio is None:
            execution.status = "REJECTED"
            execution.status_reason = "Approved at step-up, but the portfolio no longer exists."
        else:
            _execute_orders(execution, portfolio,
                            "Approved by the account holder at step-up; review anchored by Component 3.")

    session.commit()
    session.refresh(execution)
    return ExecutionResponse.model_validate(execution)


@app.post("/executions/{execution_id}/retry", response_model=ExecutionResponse)
def retry_execution(
    execution_id: str,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> ExecutionResponse:
    """Re-run the stages of a HELD execution once Component 2 or 3 is back."""
    execution = _owned_execution(execution_id, user, session)
    if execution.status != "HELD":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail=f"Only HELD executions can be retried; this one is {execution.status}")
    portfolio = session.get(Portfolio, execution.portfolio_id) if execution.portfolio_id else None
    if portfolio is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="The portfolio no longer exists")

    # Rescore only if Component 2 never answered: a second score of the same transaction
    # would count twice in its per-account history and rate limit.
    previous = execution.fraud if execution.fraud and execution.fraud.get("decision") else None
    _run_stages(execution, portfolio, user.id, device_key=None, previous_fraud=previous)
    session.commit()
    session.refresh(execution)
    return ExecutionResponse.model_validate(execution)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _run_stages(execution: Execution, portfolio: Portfolio, user_id: int,
                device_key: str | None, previous_fraud: dict | None = None) -> None:
    """Component 2, then Component 3, then holdings. Mutates `execution` in place."""
    if previous_fraud is not None:
        fraud = pipeline.Downstream(True, 200, previous_fraud)
    else:
        fraud = pipeline.score_transaction(pipeline.fraud_payload(
            execution.id, execution.kind, execution.amount, execution.currency, user_id, device_key))
    usable = fraud.reachable and fraud.status_code == 200 and bool(fraud.body)
    execution.fraud = fraud.body if usable else {
        "reachable": fraud.reachable, "status_code": fraud.status_code,
        "error": (fraud.body or {}).get("detail") or fraud.error,
    }

    audit = None
    if usable:
        audit = pipeline.anchor_decision(pipeline.audit_payload(
            execution.id, execution.kind, execution.amount, execution.source_model_version,
            fraud.body, execution.plan_hash, user_id))
        if audit.status_code == 409:
            # Component 3 already holds this record (an earlier attempt saved it but could
            # not reach the chain, or it is held for review); publish that record rather than
            # evaluating twice. Its status says whether it is still held; Component 3 versions
            # that do not report one only ever held completed records here.
            retried = pipeline.retry_anchor(execution.id)
            if retried.reachable and retried.status_code == 200 and retried.body:
                body = retried.body
                audit = pipeline.Downstream(True, 200, {
                    "status": body.get("status") or "COMPLETED", "record": body.get("record"),
                    "assurance": body.get("assurance"),
                    "blockchain": body.get("blockchain"), "message": body.get("message"),
                })
            else:
                audit = retried
        execution.audit = pipeline.audit_record(audit)

    execution.status, execution.status_reason = pipeline.decide(fraud, audit)
    if execution.status == "EXECUTED":
        _execute_orders(execution, portfolio, execution.status_reason)


def _execute_orders(execution: Execution, portfolio: Portfolio, reason: str) -> None:
    by_symbol = {h.symbol: h for h in portfolio.holdings}
    try:
        closed = pipeline.apply_orders(by_symbol, execution.orders)
    except pipeline.PlanError as exc:
        execution.status = "REJECTED"
        execution.status_reason = f"Cleared for execution, but the holdings changed since the plan: {exc}"
        return
    for symbol in closed:
        portfolio.holdings.remove(by_symbol[symbol])
    execution.status, execution.status_reason = "EXECUTED", reason


def _owned_execution(execution_id: str, user: User, session: Session) -> Execution:
    """404 for someone else's execution, for the same reason as _owned_portfolio."""
    execution = session.get(Execution, execution_id)
    if execution is None or execution.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Execution not found")
    return execution


def _plan_currency(plan: dict, fallback: str) -> str:
    """Component 1 prices a plan in its own base currency, which it reports under `currency`."""
    base = (plan.get("currency") or {}).get("base") if isinstance(plan.get("currency"), dict) else None
    if isinstance(base, str) and len(base) == 3 and base.isalpha():
        return base.upper()
    return fallback

def _owned_portfolio(portfolio_id: int, user: User, session: Session) -> Portfolio:
    """Fetch a portfolio the caller owns.

    Returns 404 -- not 403 -- when it exists but belongs to someone else. A 403 would confirm
    the id is real, letting anyone enumerate how many portfolios the platform holds.
    """
    portfolio = session.get(Portfolio, portfolio_id)
    if portfolio is None or portfolio.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Portfolio not found")
    return portfolio


def _to_holding(model: HoldingModel) -> Holding:
    return Holding(
        symbol=model.symbol.upper().strip(), quantity=model.quantity,
        current_price=model.current_price, avg_daily_volume=model.avg_daily_volume,
        cost_basis=model.cost_basis,
    )


def _to_response(portfolio: Portfolio) -> PortfolioResponse:
    return PortfolioResponse(
        id=portfolio.id, name=portfolio.name, base_currency=portfolio.base_currency,
        holdings=[HoldingModel.model_validate(h) for h in portfolio.holdings],
        total_value=portfolio.total_value,
        created_at=portfolio.created_at, updated_at=portfolio.updated_at,
    )
