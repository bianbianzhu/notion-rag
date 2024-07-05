import { createChain } from "./chains";
import { HumanMessage, AIMessage } from "@langchain/core/messages";

async function main() {
  const chain = await createChain();

  const input = "why we have this error: documents.map is not a function";

  const output = await chain.invoke({
    input,
    chat_history: [],
  });

  console.log(output.answer);
}

main();
