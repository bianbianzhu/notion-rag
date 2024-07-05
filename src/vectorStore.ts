import { NotionAPILoader } from "@langchain/community/document_loaders/web/notionapi";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { QdrantVectorStore } from "@langchain/qdrant";
import { OpenAIEmbeddings } from "@langchain/openai";
import { VectorStore } from "@langchain/core/vectorstores";

enum NotionDataSourceType {
  Page = "page",
  Database = "database",
}

export const VECTOR_STORE_COLLECTION_NAME = "iam_poor";

// Loading the Notion page with all child child pages

const pageId = process.env.NOTION_PAGE_ID ?? "";
const pageAuth = process.env.NOTION_INTEGRATION_TOKEN ?? "";
const cloudVectorStoreApiKey = process.env.QDRANT_API_KEY ?? "";
const cloudVectorStoreUrl = process.env.QDRANT_URL ?? "";

const pageLoader = new NotionAPILoader({
  id: pageId,
  clientOptions: { auth: pageAuth },
  type: NotionDataSourceType.Page,
});

const dbLoader = new NotionAPILoader({
  clientOptions: {
    auth: "<NOTION_INTEGRATION_TOKEN>",
  },
  id: "<DATABASE_ID>",
  type: "database",
  onDocumentLoaded: (current, total, currentTitle) => {
    console.log(`Loaded Page: ${currentTitle} (${current}/${total})`);
  },
  callerOptions: {
    maxConcurrency: 64, // Default value
  },
  propertiesAsHeader: true, // Prepends a front matter header of the page properties to the page contents
});

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 800,
  chunkOverlap: 100,
});

const embedding = new OpenAIEmbeddings();

async function populateVectorStore(): Promise<VectorStore> {
  const docs = await pageLoader.load();
  const splittedDocs = await splitter.splitDocuments(docs);

  const vectorStore = await QdrantVectorStore.fromDocuments(
    splittedDocs,
    embedding,
    {
      apiKey: cloudVectorStoreApiKey,
      url: cloudVectorStoreUrl,
      collectionName: VECTOR_STORE_COLLECTION_NAME,
    }
  );

  return vectorStore;
}

populateVectorStore();
