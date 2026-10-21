/**
 * Whether a nav entry is the current section. Exact match for the entry itself, plus any
 * sub-route beneath it ("/audit/decisions/EXE-1" belongs to "/audit") -- but not a sibling
 * that merely shares a prefix ("/auditor" is not "/audit").
 */
export function isNavActive(pathname: string | null, href: string): boolean {
  if (!pathname) return false;
  return pathname === href || pathname.startsWith(`${href}/`);
}
