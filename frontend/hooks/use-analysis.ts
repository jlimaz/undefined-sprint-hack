"use client";

import { create } from "zustand";

import type { LayaRole } from "@/lib/laya-files";

export type AnalysisSummary = {
  knowledge_name: string;
  data_name: string;
  observations: number;
  total_rows: number;
  questions: {
    name: string;
    instructions: string;
    counts: Record<string, number>;
  }[];
  low_confidence: number;
  seconds: number;
};

export type AnalysisError = { message: string; field: LayaRole | null };

type AnalysisState = {
  /** Laya is classifying; the chat model is unloaded until it finishes. */
  running: boolean;
  /** The classification the chat currently answers from, if any. */
  summary: AnalysisSummary | null;
  error: AnalysisError | null;
  run: (knowledge: File, data: File) => Promise<boolean>;
  clear: () => Promise<void>;
  refresh: () => Promise<void>;
};

const POLL_MS = 2000;

/** Shared by the Library, which starts a run, and the thread, which reflects it. */
export const useAnalysis = create<AnalysisState>((set, get) => ({
  running: false,
  summary: null,
  error: null,

  run: async (knowledge, data) => {
    if (get().running) return false;
    set({ running: true, error: null });
    try {
      const res = await fetch("/api/pipeline/classify", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          knowledge: { name: knowledge.name, text: await knowledge.text() },
          data: { name: data.name, text: await data.text() },
        }),
      });
      const payload = await res.json().catch(() => null);
      if (!res.ok || !payload?.summary) {
        set({
          running: false,
          error: {
            message: payload?.error ?? `The Laya service answered ${res.status}.`,
            field: payload?.field ?? null,
          },
        });
        return false;
      }
      set({ running: false, summary: payload.summary });
      return true;
    } catch {
      set({
        running: false,
        error: { message: "Could not reach the dashboard server.", field: null },
      });
      return false;
    }
  },

  clear: async () => {
    set({ summary: null, error: null });
    await fetch("/api/pipeline/context", { method: "DELETE" }).catch(() => undefined);
  },

  // Picks up a classification that outlived a page reload, or is still going.
  refresh: async () => {
    const res = await fetch("/api/pipeline/context").catch(() => null);
    if (!res?.ok) return;
    const payload = await res.json().catch(() => null);
    if (!payload) return;
    set({
      running: payload.running === true,
      summary: payload.active ? payload.summary : null,
    });
    if (payload.running) setTimeout(() => void get().refresh(), POLL_MS);
  },
}));
