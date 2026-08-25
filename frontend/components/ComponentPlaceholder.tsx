/**
 * Placeholder for a component that has not been built yet.
 *
 * These pages exist so the integration surface is visible rather than theoretical: a
 * teammate can see exactly where their component lands and what wiring it up involves,
 * without reading anyone else's code.
 */

import { Card, Notice } from "@/components/ui";

export function ComponentPlaceholder({
  component,
  title,
  owner,
  objective,
  port,
  capabilities,
}: {
  component: string;
  title: string;
  owner: string;
  objective: string;
  port: number;
  capabilities: string[];
}) {
  return (
    <div className="space-y-6">
      <div>
        <p className="text-xs uppercase tracking-wide text-neutral-500">
          Component {component}
        </p>
        <h1 className="text-xl font-semibold">{title}</h1>
        <p className="mt-1 text-sm text-neutral-500">{owner}</p>
      </div>

      <Notice tone="info" title="Not yet implemented">
        This screen is a reserved slot. The shell, auth and API client are already in place,
        so wiring this component up does not require changing anything outside its own files.
      </Notice>

      <Card title="Objective (from the TAF)">
        <p className="text-sm">{objective}</p>
      </Card>

      <Card title="Planned capabilities">
        <ul className="list-inside list-disc space-y-1 text-sm">
          {capabilities.map((capability) => (
            <li key={capability}>{capability}</li>
          ))}
        </ul>
      </Card>

      <Card title="How to wire this up">
        <ol className="list-inside list-decimal space-y-2 text-sm">
          <li>
            Serve the FastAPI app on port <code>{port}</code> and copy{" "}
            <code>service/cors.py</code> from Component 1 — the browser calls each backend
            directly, so without CORS headers every response is rejected before your code
            runs.
          </li>
          <li>
            Copy <code>service/auth.py</code> too, and share the same <code>JWT_SECRET</code>.
            Tokens are issued by the platform service and only verified by each component, so
            no service depends on another being up.
          </li>
          <li>
            Add the base URL to <code>SERVICES</code> in <code>lib/api/client.ts</code> and
            uncomment the matching env var.
          </li>
          <li>
            Add your schema to the <code>gen:api</code> script so the TypeScript types are
            generated from your OpenAPI rather than hand-written — hand-copied interfaces
            drift silently.
          </li>
          <li>
            Write <code>lib/api/{title.toLowerCase().split(" ")[0]}.ts</code> as a typed
            client, and replace this page.
          </li>
          <li>
            Flip <code>ready: true</code> for your entry in <code>NAV</code> in{" "}
            <code>app/(platform)/layout.tsx</code>.
          </li>
        </ol>
      </Card>

      <Card title="What Component 1 already gives you">
        <p className="text-sm">
          Every decision is published to the Kafka topic <code>portfolio.decisions</code> with
          the response payload and a <code>model_version</code>. The producer is
          fire-and-forget, so a consumer being down never blocks a withdrawal. Portfolios and
          identity live in the shared platform service, so you can read the same holdings
          without depending on Component 1 at all.
        </p>
      </Card>
    </div>
  );
}
