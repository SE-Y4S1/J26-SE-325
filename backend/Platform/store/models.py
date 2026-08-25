"""ORM models: users, portfolios, holdings.

Holdings are stored in exactly the shape Component 1's `Holding` contract expects
(symbol, quantity, current_price, avg_daily_volume, cost_basis), so the frontend can pass a
stored portfolio straight to /portfolio/withdraw with no translation layer. A translation
layer is where field-name drift hides, and that contract is frozen precisely because three
other components bind to it.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(120), nullable=False)
    # The hash only. A column named `password` is an invitation to put one there.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    portfolios: Mapped[list["Portfolio"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", lazy="selectin"
    )


class Portfolio(Base):
    __tablename__ = "portfolios"
    # One user cannot have two portfolios with the same name; without this the UI's
    # "rename" flow can silently create a duplicate the user then cannot tell apart.
    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_portfolio_user_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    base_currency: Mapped[str] = mapped_column(String(3), default="USD", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    owner: Mapped[User] = relationship(back_populates="portfolios")
    holdings: Mapped[list["Holding"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan", lazy="selectin"
    )

    @property
    def total_value(self) -> float:
        return sum(h.quantity * h.current_price for h in self.holdings)


class Holding(Base):
    __tablename__ = "holdings"
    __table_args__ = (UniqueConstraint("portfolio_id", "symbol", name="uq_holding_symbol"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    portfolio_id: Mapped[int] = mapped_column(
        ForeignKey("portfolios.id", ondelete="CASCADE"), index=True, nullable=False
    )

    # --- these five mirror Component 1's Holding contract exactly ---
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)
    current_price: Mapped[float] = mapped_column(Float, nullable=False)
    # Component 1's liquidity cost model divides by this, and a zero would make the asset
    # look infinitely illiquid rather than raising -- so it is required, not optional.
    avg_daily_volume: Mapped[float] = mapped_column(Float, nullable=False)
    cost_basis: Mapped[float | None] = mapped_column(Float, nullable=True)

    portfolio: Mapped[Portfolio] = relationship(back_populates="holdings")

    @property
    def value(self) -> float:
        return self.quantity * self.current_price
