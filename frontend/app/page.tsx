import { redirect } from "next/navigation";

/**
 * The landing route.
 *
 * Redirects into the platform shell, which owns the auth guard — putting the guard in one
 * place means every screen inherits it rather than each remembering to check.
 */
export default function Home() {
  redirect("/portfolio");
}
