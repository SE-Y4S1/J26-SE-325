/**
 * The Nexa mark: four nodes joined by one line, in pipeline order (plan, screen, anchor,
 * explain), tracing the letter N. The first node carries the accent, as the first stage
 * does on the landing page.
 *
 * Strokes use currentColor, so the mark follows the surrounding text colour in light and
 * dark mode. app/icon.svg is the same geometry on a fixed dark tile for the browser tab.
 */

export function NexaMark({ className = "size-7" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden focusable="false">
      <path
        d="M7 25V7l18 18V7"
        fill="none"
        stroke="currentColor"
        strokeWidth="3.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="7" cy="7" r="3.25" fill="currentColor" />
      <circle cx="25" cy="25" r="3.25" fill="currentColor" />
      <circle cx="25" cy="7" r="3.25" fill="currentColor" />
      <circle cx="7" cy="25" r="3.75" className="fill-emerald-600 dark:fill-emerald-500" />
    </svg>
  );
}

export function NexaLogo({
  markClassName = "size-7",
  textClassName = "text-lg",
}: {
  markClassName?: string;
  textClassName?: string;
}) {
  return (
    <span className="inline-flex items-center gap-2">
      <NexaMark className={markClassName} />
      <span className={`font-semibold tracking-tight ${textClassName}`}>Nexa</span>
    </span>
  );
}
