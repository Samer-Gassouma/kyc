/**
 * Dynamic API base URL — uses current hostname so it works from any network IP.
 * Backend is expected on the same host, port 8000.
 */
export const API_BASE =
  typeof window !== "undefined"
    ? `${window.location.protocol}//${window.location.hostname}:8000`
    : "http://localhost:8000";

export function getWsUrl(path: string): string {
  const host =
    typeof window !== "undefined" ? window.location.hostname : "localhost";
  return `ws://${host}:8000${path}`;
}
