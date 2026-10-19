import { redirect } from "next/navigation";

export default function ResponsibleAIRedirectPage() {
  redirect("/assistant?tab=responsible-ai");
}
