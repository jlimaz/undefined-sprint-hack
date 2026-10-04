import { AIMessage, SystemMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { ChatOllama } from "@langchain/ollama";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";
import { ToolNode } from "@langchain/langgraph/prebuilt";
import { z } from "zod";

const PIPELINE_API_URL = process.env.PIPELINE_API_URL?.trim() || "http://127.0.0.1:8000";

const OPERATOR =
  "You assist an RF signal operator. Answer in plain language, using only the " +
  "Laya classification results below. Do not recount, revise or second-guess " +
  "them. For anomalies, report the low-confidence observations and any answer " +
  "that is rare. If the results do not contain the answer, say so.";
const GROUNDED =
  OPERATOR +
  " When the operator asks to analyze a different number of rows, the next batch, " +
  "or the rows not in this result, call analyze_rows. Do not call it to answer " +
  "an ordinary question.";
const AFTER_TOOL =
  OPERATOR +
  " analyze_rows has just run. If its reply is an error, tell the operator that error " +
  "and do not describe it as a new classification. Otherwise summarize these results, " +
  "including which rows they cover.";
const NOT_LOADED =
  "You assist an RF signal operator. No signal data has been analysed yet, so " +
  "you have no results to answer from. Reply in one or two sentences: tell the " +
  "operator to add a knowledge file and a data file in the Library, press Run " +
  "Laya, and ask again. Do not invent signals or results.";
const UNREACHABLE =
  "You assist an RF signal operator. The Laya service that analyses signal " +
  "data is not reachable, so you have no results to answer from. Reply in one " +
  "or two sentences: tell the operator to start it with `npm run dev` and ask " +
  "again. Do not invent signals or results.";

type PipelineContext = { active: boolean; running: boolean; context?: string };

const model = new ChatOllama({
  // Must be the model the pipeline keeps loaded; a small GPU cannot hold two.
  model: process.env.OLLAMA_MODEL || "qwen2.5:14b",
  baseUrl: process.env.OLLAMA_HOST || "http://127.0.0.1:11434",
  streaming: true,
  numCtx: Number(process.env.OLLAMA_NUM_CTX) || undefined,
});

const analyzeRows = tool(
  async ({ count, remaining }) => {
    const body: { count?: number; remaining?: boolean } = {};
    if (typeof count === "number") body.count = count;
    if (typeof remaining === "boolean") body.remaining = remaining;
    try {
      const res = await fetch(`${PIPELINE_API_URL}/reclassify`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await res.json().catch(() => null);
      if (!res.ok || !payload?.summary) {
        return payload?.error ?? `The Laya service answered ${res.status}.`;
      }
      const summary = payload.summary;
      const start = summary.row_offset + 1;
      const end = summary.row_offset + summary.observations;
      return (
        `Classified rows ${start}–${end} ` +
        `(${summary.observations} of ${summary.total_rows}). ` +
        "Summarize this result for the operator. Do not call the tool again unless they ask for a different set of rows."
      );
    } catch {
      return "Could not reach the Laya service.";
    }
  },
  {
    name: "analyze_rows",
    description:
      "Re-run Laya on the files already loaded and replace the current result. " +
      "Call this only when the operator asks to analyze a different number of rows, " +
      "the next batch, or the rows that were not in the current result. " +
      "Do not call it to answer an ordinary question about the current result. " +
      "A bare count classifies that many rows from the start of the file. " +
      "Set remaining to true and omit count to classify every row after the current result, " +
      'which is what "analyze the remaining rows and give me another summary" means. ' +
      "Set remaining to true with a count to classify that many of those leftover rows.",
    schema: z.object({
      count: z
        .number()
        .int()
        .positive()
        .optional()
        .describe(
          "How many rows to classify. Omit it to use 100 from the start, or every leftover row when remaining is true.",
        ),
      remaining: z
        .boolean()
        .optional()
        .describe(
          "True to start after the rows in the current result. Omit or false to start from the first row.",
        ),
    }),
  },
);

const modelWithTools = model.bindTools([analyzeRows]);

async function readPipelineContext(): Promise<PipelineContext | null> {
  try {
    const res = await fetch(`${PIPELINE_API_URL}/context`, {
      signal: AbortSignal.timeout(2000),
    });
    return res.ok ? ((await res.json()) as PipelineContext) : null;
  } catch {
    return null;
  }
}

function instructions(pipeline: PipelineContext | null, afterTool: boolean) {
  if (!pipeline) return UNREACHABLE;
  if (!pipeline.active) return NOT_LOADED;
  const guidance = afterTool ? AFTER_TOOL : GROUNDED;
  return `${guidance}\n\n${pipeline.context}`;
}

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const pipeline = await readPipelineContext();
  if (pipeline?.running) {
    // Laya has the GPU; calling Ollama now would load its model back over it.
    return {
      messages: [new AIMessage("Laya is still classifying. Ask again when it finishes.")],
    };
  }
  // Read on every turn and kept out of state, so a new run replaces the old one.
  const response = await modelWithTools.invoke([
    new SystemMessage(instructions(pipeline, false)),
    ...state.messages,
  ]);
  return { messages: [response] };
};

const respond = async (state: typeof MessagesAnnotation.State) => {
  // The tool has finished, so this call is not blocked by the run it just started.
  const pipeline = await readPipelineContext();
  const response = await model.invoke([
    new SystemMessage(instructions(pipeline, true)),
    ...state.messages,
  ]);
  return { messages: [response] };
};

function requestedTool(state: typeof MessagesAnnotation.State) {
  const last = state.messages[state.messages.length - 1] as AIMessage;
  return last?.tool_calls?.length ? "tools" : "__end__";
}

export const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addNode("tools", new ToolNode([analyzeRows]))
  .addNode("respond", respond)
  .addEdge("__start__", "agent")
  .addConditionalEdges("agent", requestedTool)
  .addEdge("tools", "respond")
  .addEdge("respond", "__end__")
  .compile();
