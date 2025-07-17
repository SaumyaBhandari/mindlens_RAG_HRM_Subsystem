from typing import Dict, Any, List
import json
import uuid
from datetime import datetime
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.utilities import PythonREPL
from services.vector_service import VectorService
from services.embedding_service import EmbeddingService
from schemas import EmbeddingModel, SimilarityAlgorithm, LLMModel
import redis
import os
import asyncio


class RAGService:
    def __init__(self):
        self.vector_service = VectorService()
        self.embedding_service = EmbeddingService()
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        self.python_repl = PythonREPL()

        self.tools = [
            Tool(
                name="document_search",
                func=lambda q: asyncio.run(self._search_documents(q)),
                coroutine=self._search_documents,
                description=(
                    "Search for relevant documents based on a query. "
                    "This tool is useful for finding information within documents such as CVs, "
                    "project descriptions, technical papers, or any other textual data. "
                    "Input should be a search query string relevant to the document content."
                ),
            ),
            Tool(
                name="python_repl",
                func=self.python_repl.run,
                description=(
                    "Execute Python code for calculations or data processing. "
                    "Input should be valid Python code."
                ),
            ),
            Tool(
                name="memory_search",
                func=lambda q: asyncio.run(self._search_memory(q)),
                coroutine=self._search_memory,
                description=(
                    "Search conversation history for previous context. "
                    "Input should be a search query."
                ),
            ),
        ]

        self.agent_prompt = PromptTemplate(
            template="""Answer the question below. You have access to these tools:
                    {tools}
                    If the question requires information that might be found in documents (like CVs, project descriptions, technical data, or general knowledge stored in documents), *always* consider using the 'document_search' tool. This is your primary tool for retrieving factual information from stored documents.
                    Use the following format exactly:
                    Question: the input question you must answer
                    Thought: you should always think about what to do. If the question is about specific factual information or details that could be in a document, I should use the document_search tool. For example, if asked about a person's projects, experience, or details about a topic, I should use document_search.
                    Action: the action to take, should be one of [{tool_names}]
                    Action Input: the input to the action
                    Observation: the result of the action
                    ... (this Thought/Action/Action Input/Observation can repeat N times)
                    Thought: I now know the final answer
                    Final Answer: the final answer to the original input question
                    Previous conversation history:
                    {chat_history}
                    Begin!
                    Question: {input}
                    Thought: {agent_scratchpad}""",
            input_variables=[
                "input",
                "agent_scratchpad",
                "chat_history",
            ],
        )

    async def process_query(
        self,
        query: str,
        session_id: str = None,
        use_memory: bool = True,
        similarity_algorithm: SimilarityAlgorithm = SimilarityAlgorithm.COSINE,
        llm_model: LLMModel = LLMModel.GEMINI_FLASH_LARGE,
    ) -> Dict[str, Any]:

        if not session_id:
            session_id = str(uuid.uuid4())

        self.current_session_id = session_id
        self.current_similarity_algorithm = similarity_algorithm

        # LLM
        llm = ChatGoogleGenerativeAI(
            model=llm_model.value,
            temperature=0,
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )
        print(
            f"Using LLM: {llm.model} "
            f"| session: {session_id} "
            f"| similarity: {similarity_algorithm.value}"
        )

        # Memory
        memory = None
        if use_memory:
            memory = ConversationBufferWindowMemory(
                k=5,
                memory_key="chat_history",
                return_messages=True,
            )
            history = self._load_conversation_history(session_id)
            for entry in history:
                memory.save_context(
                    {"input": entry["input"]},
                    {"output": entry["output"]},
                )

        # Agent
        agent_final_prompt = PromptTemplate(
            template=self.agent_prompt.template,
            input_variables=[
                "input",
                "agent_scratchpad",
                "chat_history",
            ]
        )

        agent = create_react_agent(
            llm,
            self.tools,
            agent_final_prompt
        )
        agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5,
            early_stopping_method="force",
        )

        try:
            result = await agent_executor.ainvoke({"input": query})
        except Exception as e:
            print(f"[ERROR] Error processing query in AgentExecutor: {e}")
            raise

        # Persist conversation
        if use_memory:
            self._save_conversation_history(session_id, query, result["output"])

        return {
            "answer": result["output"],
            "sources": getattr(self, "_last_sources", []),
            "session_id": session_id,
        }

    async def _search_documents(self, query: str) -> str:

        try:
            embeddings = await self.embedding_service.generate_embeddings(
                [query], EmbeddingModel.GEMINI
            )
            query_embedding = embeddings[0]

            results = self.vector_service.search_similar(
                query_embedding,
                limit=5,
                algorithm=getattr(self, "current_similarity_algorithm", SimilarityAlgorithm.COSINE),
            )

            self._last_sources = [
                f"{r['filename']} (chunk {r['chunk_index']})" for r in results
            ]

            if not results:
                return "No matching documents found."

            formatted = [
                f"Document: {r['filename']}\n"
                f"Relevance: {r['score']:.3f}\n"
                f"Content: {r['text'][:500]}..."
                for r in results
            ]
            return "\n---\n".join(formatted)

        except Exception as e:
            print(f"[ERROR] Error searching documents: {e}")
            return f"Error searching documents: {e}"

    async def _search_memory(self, query: str) -> str:
        try:
            history = self._load_conversation_history(self.current_session_id)
            if not history:
                return "No previous conversation found."

            relevant = [
                entry
                for entry in history
                if query.lower() in entry["input"].lower()
                or query.lower() in entry["output"].lower()
            ]
            if not relevant:
                return "No relevant previous conversation found."

            lines = ["Previous conversation snippets:"]
            for entry in relevant[-3:]:
                lines.append(f"Q: {entry['input']}")
                lines.append(f"A: {entry['output'][:200]}...")
                lines.append("---")
            return "\n".join(lines)

        except Exception as e:
            print(f"[ERROR] Error searching memory: {e}")
            return f"Error searching memory: {e}"

    def _load_conversation_history(self, session_id: str) -> List[Dict]:
        try:
            key = f"conversation:{session_id}"
            raw = self.redis_client.get(key)
            return json.loads(raw) if raw else []
        except Exception as e:
            print(f"[WARN] Failed to load conversation history: {e}")
            return []

    def _save_conversation_history(self, session_id: str, inp: str, out: str):
        try:
            key = f"conversation:{session_id}"
            history = self._load_conversation_history(session_id)
            history.append(
                {
                    "input": inp,
                    "output": out,
                    "timestamp": str(datetime.now()),
                }
            )
            history = history[-20:]  # keep last 20
            self.redis_client.setex(key, 86400, json.dumps(history))
        except Exception as e:
            print(f"[WARN] Failed to save conversation history: {e}")