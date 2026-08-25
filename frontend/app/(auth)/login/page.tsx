"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Button, Card, Field, Input, Notice } from "@/components/ui";
import { AuthProvider, useAuth } from "@/lib/auth/context";

function LoginForm() {
  const { login } = useAuth();
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await login(email, password);
      router.push("/portfolio");
    } catch (cause) {
      // The service returns the same message for unknown-email and wrong-password on
      // purpose, so it can be shown verbatim without leaking which accounts exist.
      setError(cause instanceof Error ? cause.message : "Sign in failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="mx-auto flex min-h-dvh max-w-md flex-col justify-center p-6">
      <h1 className="mb-1 text-2xl font-semibold">Smart Finance Platform</h1>
      <p className="mb-6 text-sm text-neutral-500">J26-SE-325</p>

      <Card title="Sign in">
        <form onSubmit={handleSubmit} className="space-y-4">
          <Field label="Email">
            <Input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
          </Field>
          <Field label="Password">
            <Input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
            />
          </Field>

          {error && <Notice tone="error">{error}</Notice>}

          <Button type="submit" disabled={busy} className="w-full">
            {busy ? "Signing in…" : "Sign in"}
          </Button>
        </form>

        <p className="mt-4 text-sm text-neutral-500">
          No account?{" "}
          <Link href="/register" className="underline">
            Create one
          </Link>
        </p>
      </Card>
    </main>
  );
}

export default function LoginPage() {
  // The (auth) group has no shared layout, so each page provides its own AuthProvider.
  return (
    <AuthProvider>
      <LoginForm />
    </AuthProvider>
  );
}
