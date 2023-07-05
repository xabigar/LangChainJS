import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { TokenTextSplitter } from "langchain/text_splitter";
// import { FaissStore } from "langchain/vectorstores/faiss";
import { ChatOpenAI  } from "langchain/chat_models/openai";
import { loadQAStuffChain } from "langchain/chains";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

import 'dotenv/config'

const loader = new PDFLoader("23ChatinDESRIST.pdf");
const documents = await loader.load();

console.log("Document loaded. Start running the chain...")
// # Split the documents into chunks
const splitter = new TokenTextSplitter({
    chunkSize: 500,
    chunkOverlap: 10
});

const output = await splitter.splitDocuments(documents)

// Create embeddings and vectorstore
const docsearch = await MemoryVectorStore.fromDocuments(
    output, new OpenAIEmbeddings()
);


let query = "You have all the sections for this research paper. Now, act as an academic reviewer and assess the criterion of the following paper. For the criterion, you have to assess if it is met considering these possible results: Met, Partially met, or Not met. For the criterion, you have to mention the elaborated reason why it is met or not met and provide three text different text fragments from the article that supports the decision of the result. You have to provide the response in JSON format with the following keys: -name (contains the criteria name), -sentiment (met, partially met or not met), -comment (the reason of the results), -paragraphs (an array with the THREE text fragments from the article that support the result).```Describes the proposed artifact in adequate details, which means providing a thorough and sufficient explanation or depiction of the artifact that is being proposed. Adequate details imply that the description should be comprehensive enough to provide a clear understanding of the artifact, including its features, functions, design, materials, dimensions, and any other relevant information. The level of detail should be appropriate for the context and purpose of the proposal, ensuring that the readers or audience can form a complete picture of the artifact based on the provided description."

let results = await docsearch.similaritySearch(query,  6)

// Create LLM
const model = new ChatOpenAI({
    temperature: 0,
    model_name:"gpt-3.5-turbo",
    //, openAIApiKey: "FILL"
});
// Document QA
// Create QA chain

// const chain = RetrievalQAChain.fromLLM(model, results);


console.log("Retrieving...")

const chainA = loadQAStuffChain(model);

let res = await chainA.call({
    input_documents: results,
    question: query,
});


/* const res = await chain.call({
    query: query,
    verbose:true,
});
*/
console.log({ res });