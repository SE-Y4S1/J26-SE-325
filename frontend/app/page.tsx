import Link from "next/link";

import { NexaLogo } from "@/components/NexaLogo";

/**
 * Public landing page: what Nexa is, before anyone signs in.
 *
 * The signed-in screens keep their own auth guard in app/(platform)/layout.tsx, so this
 * page needs none. Every claim here describes what the code does today; the caveats that
 * matter (simulated ledger, untrained fraud heuristic) are stated, not hidden.
 */

const STAGES = [
  {
    verb: "Plan",
    owner: "Portfolio optimization",
    body: "A fuzzy genetic algorithm sells what raises the cash for the lowest realized loss.",
  },
  {
    verb: "Screen",
    owner: "Fraud detection",
    body: "Behavioural and graph models score the transaction. Deterministic rules choose the action.",
  },
  {
    verb: "Anchor",
    owner: "Blockchain auditability",
    body: "A hash of the decision and the plan is committed, so any later change is detectable.",
  },
  {
    verb: "Explain",
    owner: "Agentic assistant",
    body: "A local language model answers why, using each component's own evidence.",
  },
] as const;

const OUTCOMES = [
  {
    status: "Executed",
    tone: "text-emerald-700 dark:text-emerald-400",
    body: "Fraud screening allowed it and the decision was anchored. Holdings change.",
  },
  {
    status: "Blocked",
    tone: "text-neutral-900 dark:text-neutral-100",
    body: "The fraud gateway or audit policy rejected it. Nothing moves, and the block is recorded.",
  },
  {
    status: "Verification required",
    tone: "text-neutral-900 dark:text-neutral-100",
    body: "A step-up was triggered. The account holder re-authenticates before anything moves.",
  },
  {
    status: "Held",
    tone: "text-neutral-900 dark:text-neutral-100",
    body: "A service was unreachable. Nexa fails closed and lets you retry once it is back.",
  },
] as const;

const TEAM = [
  { name: "Nivakaran S.", role: "Portfolio optimization and platform" },
  { name: "Dhushanthini R.", role: "Fraud detection" },
  { name: "Abeysekara W.C.S.M.", role: "Blockchain auditability" },
  { name: "W.V.A.D.K. Chamara", role: "Explainable agentic assistant" },
] as const;

export default function Home() {
  return (
    <div className="min-h-dvh">
      <header className="border-b border-black/10 dark:border-white/15">
        <nav className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-6 px-4">
          <Link href="/" aria-label="Nexa home">
            <NexaLogo />
          </Link>
          <div className="flex items-center gap-1 text-sm sm:gap-2">
            <a href="#how-it-works" className="hidden rounded-lg px-3 py-2 text-neutral-600 hover:text-neutral-900 sm:block dark:text-neutral-400 dark:hover:text-neutral-100">
              How it works
            </a>
            <a href="#components" className="hidden rounded-lg px-3 py-2 text-neutral-600 hover:text-neutral-900 sm:block dark:text-neutral-400 dark:hover:text-neutral-100">
              Components
            </a>
            <Link href="/login" className="rounded-lg px-3 py-2 font-medium hover:bg-black/5 dark:hover:bg-white/10">
              Sign in
            </Link>
            <PrimaryLink href="/register">Create account</PrimaryLink>
          </div>
        </nav>
      </header>

      <main>
        <Hero />
        <Components />
        <Outcomes />
        <Principle />
        <Team />
        <ClosingCta />
      </main>

      <footer className="border-t border-black/10 dark:border-white/15">
        <div className="mx-auto flex max-w-6xl flex-col gap-2 px-4 py-8 text-sm text-neutral-500 sm:flex-row sm:items-center sm:justify-between">
          <p>Nexa is the J26-SE-325 research project, SLIIT IT4010.</p>
          <p>A research prototype. Balances are simulated, not real brokerage holdings.</p>
        </div>
      </footer>
    </div>
  );
}

function Hero() {
  return (
    <section className="mx-auto grid max-w-6xl gap-12 px-4 pt-16 pb-20 md:grid-cols-[1.1fr_1fr] md:items-center md:pt-24">
      <div className="nexa-rise">
        <h1 className="text-4xl font-semibold leading-[1.05] tracking-tighter md:text-5xl lg:text-6xl">
          Portfolio decisions you can screen, prove and explain.
        </h1>
        <p className="mt-6 max-w-[48ch] text-lg leading-relaxed text-neutral-600 dark:text-neutral-400">
          Nexa plans liquidity-aware withdrawals, checks every transaction for fraud, anchors
          decisions on a ledger and explains why.
        </p>
        <div className="mt-8 flex flex-wrap gap-3">
          <PrimaryLink href="/register" large>
            Create account
          </PrimaryLink>
          <Link
            href="/login"
            className="rounded-lg border border-black/15 px-5 py-3 text-sm font-medium transition-[background-color,transform] duration-200 hover:bg-black/5 active:translate-y-px dark:border-white/20 dark:hover:bg-white/10"
          >
            Sign in
          </Link>
        </div>
      </div>

      <ol
        id="how-it-works"
        aria-label="How one plan moves through Nexa"
        className="nexa-rise nexa-delay relative scroll-mt-24 space-y-0 rounded-2xl border border-black/10 bg-neutral-50 p-6 dark:border-white/15 dark:bg-neutral-900"
      >
        {STAGES.map((stage, index) => (
          <li key={stage.verb} className="relative flex gap-4 pb-6 last:pb-0">
            {index < STAGES.length - 1 && (
              <span
                aria-hidden
                className="absolute top-9 left-[15px] h-[calc(100%-2.25rem)] w-px bg-black/15 dark:bg-white/20"
              />
            )}
            <span
              aria-hidden
              className={`mt-1 flex size-8 shrink-0 items-center justify-center rounded-full text-xs font-semibold ${
                index === 0
                  ? "bg-emerald-600 text-white"
                  : "border border-black/15 bg-white text-neutral-600 dark:border-white/20 dark:bg-neutral-950 dark:text-neutral-300"
              }`}
            >
              {index + 1}
            </span>
            <div>
              <p className="font-semibold">
                {stage.verb}
                <span className="ml-2 text-sm font-normal text-neutral-500">{stage.owner}</span>
              </p>
              <p className="mt-1 text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
                {stage.body}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

function Components() {
  return (
    <section id="components" className="scroll-mt-16 border-t border-black/10 dark:border-white/15">
      <div className="mx-auto max-w-6xl px-4 py-20">
        <h2 className="max-w-[24ch] text-3xl font-semibold tracking-tight md:text-4xl">
          Four research components, one product.
        </h2>
        <p className="mt-4 max-w-[60ch] leading-relaxed text-neutral-600 dark:text-neutral-400">
          Each runs as its own service with its own models and tests. The platform connects
          them, so a plan is never executed without being screened and recorded.
        </p>

        <div className="mt-12 grid gap-4 md:grid-cols-3 md:grid-rows-2">
          <article className="rounded-2xl bg-emerald-600 p-8 text-white md:col-span-2 md:row-span-2 md:flex md:flex-col md:justify-between">
            <div>
              <h3 className="text-2xl font-semibold tracking-tight">Liquidity-aware portfolio optimization</h3>
              <p className="mt-4 max-w-[52ch] leading-relaxed text-emerald-50">
                Instant withdrawals planned by a fuzzy inference system and genetic algorithm,
                compared against naive baselines. Long-term allocation with MOEA/D, sized in
                the holding&apos;s own currency.
              </p>
            </div>
            <p className="mt-8 text-sm text-emerald-100">
              Stocks on CSE and US markets, crypto on Binance, forex conversion to one base currency.
            </p>
          </article>

          <article className="rounded-2xl border border-black/10 p-6 dark:border-white/15">
            <h3 className="font-semibold">Real-time fraud detection</h3>
            <p className="mt-2 text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
              An LSTM autoencoder and a graph network score each transaction. Allow, step-up or
              block is chosen by fixed rules, never by a model.
            </p>
          </article>

          <article className="rounded-2xl bg-neutral-900 p-6 text-neutral-100 dark:bg-neutral-100 dark:text-neutral-900">
            <h3 className="font-semibold">Blockchain auditability</h3>
            <p className="mt-2 text-sm leading-relaxed text-neutral-300 dark:text-neutral-700">
              Only a SHA-256 commitment goes on-chain, so amounts stay private. A simulated
              ledger stands in when no chain is connected.
            </p>
          </article>

          <article className="rounded-2xl border border-black/10 bg-neutral-50 p-6 md:col-span-3 dark:border-white/15 dark:bg-neutral-900">
            <h3 className="font-semibold">Explainable agentic assistant</h3>
            <p className="mt-2 max-w-[80ch] text-sm leading-relaxed text-neutral-600 dark:text-neutral-400">
              A LangGraph agent on a local Ollama model, so no data leaves the machine. Ask it
              about any execution and it reads the plan&apos;s rule trace, the fraud decision and
              the audit verification, then answers with the evidence attached.
            </p>
          </article>
        </div>
      </div>
    </section>
  );
}

function Outcomes() {
  return (
    <section className="bg-neutral-50 dark:bg-neutral-900">
      <div className="mx-auto max-w-6xl px-4 py-20">
        <h2 className="max-w-[28ch] text-3xl font-semibold tracking-tight md:text-4xl">
          Every transaction ends in one clear state.
        </h2>
        <dl className="mt-12 grid gap-x-12 gap-y-10 sm:grid-cols-2">
          {OUTCOMES.map((outcome) => (
            <div key={outcome.status}>
              <dt className={`text-xl font-semibold ${outcome.tone}`}>{outcome.status}</dt>
              <dd className="mt-2 max-w-[44ch] leading-relaxed text-neutral-600 dark:text-neutral-400">
                {outcome.body}
              </dd>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}

function Principle() {
  return (
    <section className="mx-auto max-w-4xl px-4 py-24 text-center">
      <p className="text-3xl font-semibold leading-tight tracking-tight md:text-4xl">
        Models produce scores. Rules make decisions.{" "}
        <span className="text-emerald-700 dark:text-emerald-400">People can check both.</span>
      </p>
      <p className="mx-auto mt-6 max-w-[60ch] leading-relaxed text-neutral-600 dark:text-neutral-400">
        Every response is tagged with the model that produced it. When a fraud score comes from
        an untrained fallback instead of a benchmarked model, Nexa says so.
      </p>
    </section>
  );
}

function Team() {
  return (
    <section className="border-t border-black/10 dark:border-white/15">
      <div className="mx-auto max-w-6xl px-4 py-16">
        <h2 className="text-xl font-semibold">Built by the J26-SE-325 team</h2>
        <ul className="mt-8 grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {TEAM.map((member) => (
            <li key={member.name}>
              <p className="font-medium">{member.name}</p>
              <p className="mt-1 text-sm text-neutral-500">{member.role}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}

function ClosingCta() {
  return (
    <section className="mx-auto max-w-6xl px-4 pb-24">
      <div className="flex flex-col items-start gap-6 rounded-2xl border border-black/10 p-8 sm:flex-row sm:items-center sm:justify-between md:p-12 dark:border-white/15">
        <div>
          <h2 className="text-2xl font-semibold tracking-tight">See a withdrawal go end to end.</h2>
          <p className="mt-2 text-neutral-600 dark:text-neutral-400">
            Create a portfolio, plan a withdrawal and execute it.
          </p>
        </div>
        <PrimaryLink href="/register" large>
          Create account
        </PrimaryLink>
      </div>
    </section>
  );
}

function PrimaryLink({ href, children, large = false }: { href: string; children: React.ReactNode; large?: boolean }) {
  return (
    <Link
      href={href}
      className={`shrink-0 rounded-lg bg-neutral-900 font-medium whitespace-nowrap text-white transition-[background-color,transform] duration-200 hover:bg-neutral-700 active:translate-y-px dark:bg-white dark:text-neutral-900 dark:hover:bg-neutral-200 ${
        large ? "px-5 py-3 text-sm" : "px-3 py-2 text-sm"
      }`}
    >
      {children}
    </Link>
  );
}
