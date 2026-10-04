import { AIMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOllama } from "@langchain/ollama";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";

const PIPELINE_API_URL = process.env.PIPELINE_API_URL?.trim() || "http://127.0.0.1:8000";

const GROUNDED =
  "You assist an RF signal operator. Answer in plain language, using only the " +
  "Laya classification results below. Do not recount, revise or second-guess " +
  "them. For anomalies, report the low-confidence observations and any answer " +
  "that is rare. If the results do not contain the answer, say so.";
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
  model: process.env.OLLAMA_MODEL || "deepseek-r1:14b",
  baseUrl: process.env.OLLAMA_HOST || "http://127.0.0.1:11434",
  streaming: true,
  // Thinking models reason by default. Keep the reply in the visible message.
  think: false,
  numCtx: Number(process.env.OLLAMA_NUM_CTX) || undefined,
});

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

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const pipeline = await readPipelineContext();
  if (pipeline?.running) {
    // Laya has the GPU; calling Ollama now would load its model back over it.
    return {
      messages: [new AIMessage("Laya is still classifying. Ask again when it finishes.")],
    };
  }
  // Read on every turn and kept out of state, so a new run replaces the old one.
  const system = !pipeline
    ? UNREACHABLE
    : pipeline.active
      ? `${GROUNDED}\n\n${pipeline.context}`
      : NOT_LOADED;
  const response = await model.invoke([new SystemMessage(system), ...state.messages]);
  return { messages: [response] };
};

export const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent")
  .addEdge("agent", "__end__")
  .compile();
