import { ChatOllama } from "@langchain/ollama";
import { MessagesAnnotation, StateGraph } from "@langchain/langgraph";

const model = new ChatOllama({
  model: process.env.OLLAMA_MODEL || "deepseek-r1:14b",
  baseUrl: process.env.OLLAMA_HOST || "http://127.0.0.1:11434",
  streaming: true,
  // DeepSeek R1 thinks by default. Keep the reply in the visible message.
  think: false,
});

const callModel = async (state: typeof MessagesAnnotation.State) => {
  const response = await model.invoke(state.messages);
  return { messages: [response] };
};

export const graph = new StateGraph(MessagesAnnotation)
  .addNode("agent", callModel)
  .addEdge("__start__", "agent")
  .addEdge("agent", "__end__")
  .compile();
