import { redirect } from "next/navigation";

/**
 * Root page - redirects to /explore (main dashboard).
 * In production, auth middleware checks the session before this runs.
 */
export default function Home() {
  redirect("/explore");
}
