"use client";

import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useStreamRuntime } from "@assistant-ui/react-langchain";

import { Thread } from "@/components/assistant-ui/elements/thread.aui";
import { Library } from "@/components/library";

export function Assistant() {
  const apiUrl =
    process.env.NEXT_PUBLIC_LANGGRAPH_API_URL || "http://localhost:2024";

  const runtime = useStreamRuntime({
    assistantId: process.env.NEXT_PUBLIC_LANGGRAPH_ASSISTANT_ID || "agent",
    apiUrl,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-full min-h-0 flex-col md:flex-row">
        <Library />
        <div className="min-h-0 min-w-0 flex-1">
          <Thread />
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
}
