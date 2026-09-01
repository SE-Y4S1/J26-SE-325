"use client";

/**
 * Component 4 — Localized Explainable Agentic LLM Assistant (W V A D K Chamara).
 *
 * The agent, its tools and the responsible-AI checks are entirely his. This screen sends a
 * message and renders what comes back.
 *
 * The evidence trail is shown rather than tucked away behind the answer: an explanation with
 * no tool output behind it is exactly the thing his component exists to avoid, so the UI
 * makes its absence visible instead of hiding it.
 */

import { useState } from "react";

import { Button, Card, Field, Input, Notice, Spinner } from "@/components/ui";
import { askAssistant, type AssistantReply } from "@/lib/api/assistant";
import { ApiError } from "@/lib/api/client";

export default function AssistantPage() {
  const [message, setMessage] = useState("Why was my withdrawal plan spread over three days?");
  const [reply, setReply] = useState<AssistantReply | null>(null);
  const [busy, setBusy] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAsk() {
    setBusy(true);
    setError(null);
    setUnavailable(false);
    try {
      setReply(await askAssistant(message));
    } catch (cause) {
      setReply(null);
      if (cause instanceof ApiError && (cause.isUnavailable || cause.status === 0)) {
        setUnavailable(true);
      } else {
        setError(cause instanceof Error ? cause.message : "The assistant could not answer");
      }
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-semibold">Assistant</h1>
        <p className="mt-1 text-sm text-neutral-500">
          Explains platform decisions in plain language, grounded in the evidence its tools
          return. Component 4 · W V A D K Chamara.
        </p>
      </div>

      <Card title="Ask" subtitle="Questions about a decision the platform has made.">
        <Field label="Message">
          <Input
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !busy) void handleAsk();
            }}
          />
        </Field>
        <div className="mt-4">
          <Button onClick={handleAsk} disabled={busy || !message.trim()}>
            {busy ? "Thinking…" : "Ask"}
          </Button>
        </div>
      </Card>

      {busy && <Spinner label="The agent is working" />}

      {unavailable && (
        <Notice tone="info" title="Assistant is not running">
          Start it with <code>docker compose up component4</code>, or{" "}
          <code>uvicorn main:app --port 8003</code> from{" "}
          <code>backend/Agentic-Assistance</code>. It also needs an LLM key — without one the
          service starts and reports healthy, but cannot answer.
        </Notice>
      )}

      {error && (
        <Notice tone="error" title="Could not answer">
          {error}
        </Notice>
      )}

      {reply && (
        <div className="space-y-4">
          <Card title="Answer">
            <p className="whitespace-pre-wrap text-sm">{reply.answer}</p>
          </Card>

          <Card
            title="Evidence"
            subtitle="Which tools ran, and what they returned. An answer with no evidence behind it is one to distrust."
          >
            {reply.evidence.length === 0 ? (
              <p className="text-sm text-neutral-500">
                No tools were called — the agent answered from the question alone (
                {reply.agent_execution.status}).
              </p>
            ) : (
              <ul className="space-y-3">
                {reply.evidence.map((item, index) => (
                  <li key={index} className="rounded border border-neutral-200/60 p-3">
                    <p className="text-xs font-semibold uppercase tracking-wide text-neutral-500">
                      {item.tool}
                    </p>
                    <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-xs">
                      {item.result}
                    </pre>
                  </li>
                ))}
              </ul>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
