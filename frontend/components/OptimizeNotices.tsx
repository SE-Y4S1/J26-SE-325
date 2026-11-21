import { Notice } from "@/components/ui";

/**
 * What the optimizer wants the reader to know before acting on its weights.
 *
 * `warnings` are statements the answer rests on: no forecasts (so the expected return is 0 by
 * assumption), or a trade that is a large share of a name's volume. `scenarioSource` says
 * whether the risk figure came from the holdings' own price history or from an assumed
 * correlation. Both fields are optional because an older service omits them.
 */
export function OptimizeNotices({
  warnings,
  scenarioSource,
}: {
  warnings?: string[] | null;
  scenarioSource?: string | null;
}) {
  const hasWarnings = Boolean(warnings && warnings.length > 0);
  const assumedCorrelation = scenarioSource === "assumed_correlation";
  if (!hasWarnings && !assumedCorrelation) return null;

  return (
    <div className="space-y-3">
      {hasWarnings && (
        <Notice tone="warn" title="Read this before acting on these weights">
          <ul className="list-disc space-y-1 pl-5">
            {warnings!.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </Notice>
      )}
      {assumedCorrelation && (
        <Notice tone="info" title="Risk estimated with an assumed correlation">
          There is not enough shared price history for these holdings yet, so the CVaR figure assumes
          a constant 0.5 correlation between them rather than measuring it.
        </Notice>
      )}
    </div>
  );
}
