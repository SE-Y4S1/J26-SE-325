import { useEffect, useState } from "react";
import { ApiError, getHealth, type Health } from "./api";
import { Adversarial } from "./pages/Adversarial";
import { History } from "./pages/History";
import { Live } from "./pages/Live";
import { Overview } from "./pages/Overview";
import { Research } from "./pages/Research";

const PAGES = [
  ["overview", "Overview"],
  ["live", "Live detection"],
  ["history", "History"],
  ["research", "Benchmarks"],
  ["adversarial", "Adversarial results"],
] as const;

type PageId = (typeof PAGES)[number][0];

export function App() {
  const [page, setPage] = useState<PageId>("overview");
  const [health, setHealth] = useState<Health | null>(null);
  const [offline, setOffline] = useState(false);

  useEffect(() => {
    getHealth()
      .then((body) => {
        setHealth(body);
        setOffline(false);
      })
      .catch((reason: unknown) => {
        setHealth(null);
        setOffline(true);
        if (!(reason instanceof ApiError) && !(reason instanceof TypeError)) setOffline(true);
      });
  }, [page]);

  return (
    <div className="app">
      <aside className="sidebar">
        <p className="brand">Component 02</p>
        <p className="muted">Fraud and behavioural anomaly engine</p>
        <nav>
          {PAGES.map(([id, label]) => (
            <button key={id} type="button" aria-current={page === id ? "page" : undefined} onClick={() => setPage(id)}>
              {label}
            </button>
          ))}
        </nav>
        <p className={offline ? "status down" : "status ok"}>
          {offline ? "API unreachable" : health ? `API ${health.model_version}` : "Checking API…"}
        </p>
      </aside>
      <main className="main">
        {page === "overview" ? <Overview /> : null}
        {page === "live" ? <Live /> : null}
        {page === "history" ? <History /> : null}
        {page === "research" ? <Research /> : null}
        {page === "adversarial" ? <Adversarial /> : null}
      </main>
    </div>
  );
}
