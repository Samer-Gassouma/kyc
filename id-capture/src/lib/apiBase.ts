/**
 * API base URL — same-origin, no port.
 * In production, nginx proxies /api/* and /ws/* to the backend.
 * In dev (next dev), Next.js rewrites proxy /api/* to localhost:8000.
 */
export const API_BASE =
  typeof window !== "undefined" ? "" : "http://localhost:8000";

export function getWsUrl(path: string): string {
  if (typeof window === "undefined") return `ws://localhost:8000${path}`;
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}${path}`;
}
