import { useState, type FormEvent } from "react";
import { ApiError, scoreTransaction, type ScoreResponse } from "../api";

const empty = {
  transaction_id: "",
  amount: "",
  currency: "USD",
  product_code: "",
  account_key: "",
  device_key: "",
  ip_reputation: "",
  typing_interval_ms: "",
  pointer_jitter: "",
  session_duration_s: "",
};

export function Live() {
  const [form, setForm] = useState(empty);
  const [error, setError] = useState<string | null>(null);
  const [pending, setPending] = useState(false);
  const [result, setResult] = useState<ScoreResponse | null>(null);

  function update(key: keyof typeof empty, value: string) {
    setForm((current) => ({ ...current, [key]: value }));
  }

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setResult(null);
    if (!/^[A-Za-z0-9_.:-]+$/.test(form.transaction_id)) {
      setError("Transaction ID must use letters, numbers, and . _ : - only.");
      return;
    }
    const amount = Number(form.amount);
    if (!Number.isFinite(amount) || amount < 0) {
      setError("Amount must be a number of zero or more.");
      return;
    }
    let reputation: number | undefined;
    if (form.ip_reputation.trim()) {
      reputation = Number(form.ip_reputation);
      if (!Number.isFinite(reputation) || reputation < 0 || reputation > 1) {
        setError("Reputation must be between 0 and 1. Leave it blank if you do not have a score.");
        return;
      }
    }
    const simulated = simulatedPayload(form);
    setPending(true);
    try {
      const scored = await scoreTransaction({
        transaction_id: form.transaction_id,
        amount,
        currency: form.currency || "USD",
        product_code: form.product_code || undefined,
        account_key: form.account_key || undefined,
        device_key: form.device_key || undefined,
        ip_reputation: reputation,
        simulated_signals: simulated,
      });
      setResult(scored);
    } catch (reason) {
      setResult(null);
      setError(reason instanceof ApiError ? reason.message : "The fraud API is not reachable. No decision was produced.");
    } finally {
      setPending(false);
    }
  }

  return (
    <section>
      <h1>Live fraud detection</h1>
      <p className="muted">
        The form posts one transaction to the existing gateway. Behavioural fields are sent only as simulated signals. The API rejects names, card numbers, and raw network addresses.
      </p>
      <form className="stack" onSubmit={onSubmit}>
        <div className="row">
          <label>
            Transaction ID
            <input value={form.transaction_id} onChange={(event) => update("transaction_id", event.target.value)} required />
          </label>
          <label>
            Amount
            <input value={form.amount} onChange={(event) => update("amount", event.target.value)} inputMode="decimal" required />
          </label>
        </div>
        <div className="row">
          <label>
            Currency
            <input value={form.currency} maxLength={3} onChange={(event) => update("currency", event.target.value.toUpperCase())} />
          </label>
          <label>
            Product code
            <input value={form.product_code} maxLength={8} onChange={(event) => update("product_code", event.target.value)} />
          </label>
        </div>
        <div className="row">
          <label>
            Account key
            <input value={form.account_key} onChange={(event) => update("account_key", event.target.value)} />
          </label>
          <label>
            Device key
            <input value={form.device_key} onChange={(event) => update("device_key", event.target.value)} />
          </label>
        </div>
        <label>
          Supplied reputation, 0 worst to 1 trusted
          <input value={form.ip_reputation} onChange={(event) => update("ip_reputation", event.target.value)} placeholder="optional" />
        </label>
        <fieldset>
          <legend>Simulated behavioural signals</legend>
          <div className="row">
            <label>
              Typing interval (ms)
              <input value={form.typing_interval_ms} onChange={(event) => update("typing_interval_ms", event.target.value)} />
            </label>
            <label>
              Pointer jitter
              <input value={form.pointer_jitter} onChange={(event) => update("pointer_jitter", event.target.value)} />
            </label>
            <label>
              Session duration (s)
              <input value={form.session_duration_s} onChange={(event) => update("session_duration_s", event.target.value)} />
            </label>
          </div>
        </fieldset>
        <button className="primary" type="submit" disabled={pending}>
          {pending ? "Scoring…" : "Score transaction"}
        </button>
      </form>
      {error ? <p className="error" role="alert">{error}</p> : null}
      {result ? <ResultCard result={result} /> : null}
    </section>
  );
}

function simulatedPayload(form: typeof empty) {
  const typing = optionalNumber(form.typing_interval_ms);
  const jitter = optionalNumber(form.pointer_jitter);
  const session = optionalNumber(form.session_duration_s);
  if (typing === undefined && jitter === undefined && session === undefined) return undefined;
  return {
    origin: "simulated" as const,
    typing_interval_ms: typing,
    pointer_jitter: jitter,
    session_duration_s: session,
  };
}

function optionalNumber(value: string): number | undefined {
  if (!value.trim()) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
}

function ResultCard({ result }: { result: ScoreResponse }) {
  return (
    <article className={`banner ${result.decision}`} aria-live="polite">
      <p className="muted">Gateway decision</p>
      <h2 className={result.decision}>{result.decision}</h2>
      <p>Fraud score {result.fraud_score.toFixed(4)}</p>
      <p className="scope">
        Model {result.model_version}. Benchmark model: {result.benchmark_model ? "yes" : "no"}.
      </p>
      <p>Rules: {result.explanation.rules.join(", ") || "none returned"}</p>
      {Object.keys(result.explanation.streams).length ? (
        <p>
          Streams:{" "}
          {Object.entries(result.explanation.streams)
            .map(([name, value]) => `${name} ${value.toFixed(4)}`)
            .join(", ")}
        </p>
      ) : null}
      {result.explanation.notes.length ? <p>{result.explanation.notes.join(" ")}</p> : null}
      {result.explanation.top_factors.length ? (
        <p>
          Largest logistic contributions:{" "}
          {result.explanation.top_factors.map((factor) => `${factor.name} (${factor.contribution.toFixed(3)})`).join(", ")}
        </p>
      ) : null}
    </article>
  );
}
