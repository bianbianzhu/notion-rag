import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAIEmbeddings } from "@langchain/openai";
import { VECTOR_STORE_COLLECTION_NAME } from "./vectorStore";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import chatModel from "./chatModel";

const K_VALUE = 4;

const systemFStringTemplate: string = `You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Context: {context}
`;

const contextualizeQSystemFStringTemplate: string = `Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.`;

const chatPromptTemplate = ChatPromptTemplate.fromMessages([
  ["system", systemFStringTemplate],
  ["placeholder", "{chat_history}"],
  ["user", "Question: {input}"],
]);

const contextualizeQChatPromptTemplate = ChatPromptTemplate.fromMessages([
  ["placeholder", "{chat_history}"],
  ["user", "Question: {input}"],
  ["system", contextualizeQSystemFStringTemplate],
]);

export async function createChain() {
  const retriever = await createRetriever();

  const historyAwareRetriever = await createHistoryAwareRetriever({
    llm: chatModel,
    retriever,
    rephrasePrompt: contextualizeQChatPromptTemplate,
  });

  const documentChain = await createStuffDocumentsChain({
    llm: chatModel,
    prompt: chatPromptTemplate,
    outputParser: new StringOutputParser(),
  });

  const conversationChain = await createRetrievalChain({
    retriever: historyAwareRetriever,
    combineDocsChain: documentChain,
  });

  return conversationChain;
}

async function createRetriever() {
  const cloudVectorStoreUrl = process.env.QDRANT_URL ?? "";

  const embedding = new OpenAIEmbeddings();
  const vectorStore = await QdrantVectorStore.fromExistingCollection(
    embedding,
    {
      url: cloudVectorStoreUrl,
      collectionName: VECTOR_STORE_COLLECTION_NAME,
    }
  );

  const retriever = vectorStore.asRetriever(K_VALUE);

  return retriever;
}
