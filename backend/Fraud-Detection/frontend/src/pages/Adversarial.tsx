import { RESEARCH_STUDIES, formatMetric, type AttackResult, type Study } from "../studies";

const LABELS: Record<string, string> = {
  camouflage: "Camouflage",
  slow_drift: "Slow drift",
  structuring: "Structuring",
};

export function Adversarial() {
  const studies = RESEARCH_STUDIES.filter((study) => study.attacks.length > 0);
  const missing = RESEARCH_STUDIES.filter((study) => study.attacks.length === 0);

  return (
    <section>
      <h1>Adversarial and drift results</h1>
      <p className="muted">
        This page shows attacks that were already executed and saved. It does not launch a new attack. There is no API endpoint that reruns camouflage, drift, or structuring.
      </p>
      <p className="notice">
        To reproduce an executed study from the Fraud-Detection folder: <code>py scripts\train_benchmark.py --profile dev --dataset ieee</code> or <code>--profile wider --dataset ieee</code>.
      </p>
      {studies.map((study) => (
        <article className="card" key={study.id} style={{ marginTop: "0.85rem" }}>
          <h2>{study.title}</h2>
          <p className="scope">
            Scored model: {study.attackModel ?? "not recorded"}. Research benchmark: {study.research ? "yes" : "no"}.
          </p>
          <table>
            <thead>
              <tr>
                <th>Attack</th>
                <th>Clean AUROC</th>
                <th>Attack AUROC</th>
                <th>Change</th>
                <th>Gateway effect</th>
              </tr>
            </thead>
            <tbody>
              {study.attacks.map((attack) => (
                <tr key={attack.name}>
                  <td>{LABELS[attack.name] ?? attack.name}</td>
                  <td>{formatMetric(attack.cleanAuroc)}</td>
                  <td>{formatMetric(attack.auroc)}</td>
                  <td>{formatMetric(attack.change)}</td>
                  <td>{effect(attack)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </article>
      ))}
      {missing.map((study: Study) => (
        <article className="card" key={study.id} style={{ marginTop: "0.85rem" }}>
          <h2>{study.title}</h2>
          <p>This saved file does not contain camouflage, slow-drift, or structuring results.</p>
        </article>
      ))}
    </section>
  );
}

function effect(attack: AttackResult): string {
  if (attack.fraudStillBlocked !== null && attack.fraudRows !== null) {
    return `${attack.fraudStillBlocked} of ${attack.fraudRows} fraud rows still blocked`;
  }
  if (attack.fullyAllowed !== null && attack.cases !== null) {
    const blocked = attack.anyPartBlocked === null ? "not measured" : String(attack.anyPartBlocked);
    return `${attack.fullyAllowed} of ${attack.cases} cases fully allowed; any part blocked: ${blocked}; parts: ${attack.parts ?? "not recorded"}`;
  }
  return "not measured";
}
