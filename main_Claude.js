import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { ChatAnthropic } from "langchain/chat_models/anthropic";
import { loadQAStuffChain } from "langchain/chains";

import dotenv from 'dotenv'
dotenv.config()
console.log(process.env.OPENAI_API_KEY)


const loader = new PDFLoader("23ChatinDESRIST.pdf");

const documents = await loader.load();

// console.log(documents)
console.log("Document loaded. Start running the chain...")

// Create LLM
const model = new ChatAnthropic({
    temperature: 0,
});
// Document QA
let query = "You have all the sections for this research paper. Now, act as an academic reviewer and assess the criterion of the following paper. For the criterion, you have to assess if it is met considering these possible results: Met, Partially met, or Not met. For the criterion, you have to mention the elaborated reason why it is met or not met and provide three text different text fragments from the article that supports the decision of the result. You have to provide the response in JSON format with the following keys: -name (contains the criteria name), -sentiment (met, partially met or not met), -comment (the reason of the results), -paragraphs (an array with the THREE text fragments from the article that support the result).```Describes the proposed artifact in adequate details, which means providing a thorough and sufficient explanation or depiction of the artifact that is being proposed. Adequate details imply that the description should be comprehensive enough to provide a clear understanding of the artifact, including its features, functions, design, materials, dimensions, and any other relevant information. The level of detail should be appropriate for the context and purpose of the proposal, ensuring that the readers or audience can form a complete picture of the artifact based on the provided description."
// Create QA chain
const chain = loadQAStuffChain(model);
let res = await chain.call({
    input_documents: documents,
    question: query,
})

console.log({ res });
