from __future__ import annotations

from dataclasses import dataclass

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.agent.graph import KnowledgeAgent
from app.agent.nodes import AgentDependencies
from app.agent.tools import InternalApiClient, ToolPlanner, build_default_registry
from app.api.schemas import UserContext
from app.core.config import Settings
from app.db.session import create_engine_and_session
from app.rag.bm25 import BM25Index
from app.rag.chunker import StructureAwareChunker
from app.rag.citation import CitationBuilder
from app.rag.embedder import EmbeddingService
from app.rag.hybrid import HybridRetriever
from app.rag.parser import DocumentParser
from app.rag.query_rewrite import QueryRewriter
from app.rag.reranker import Reranker
from app.rag.vector_store import MilvusVectorStore
from app.services.conversation_service import ConversationService
from app.services.evaluation_service import EvaluationService
from app.services.knowledge_service import KnowledgeService
from app.services.permission_service import PermissionService


@dataclass
class Container:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    conversation: ConversationService
    permission: PermissionService
    embedder: EmbeddingService
    vector_store: MilvusVectorStore
    bm25: BM25Index
    reranker: Reranker
    rewriter: QueryRewriter
    retriever: HybridRetriever
    knowledge: KnowledgeService
    evaluation: EvaluationService
    agent: KnowledgeAgent
    llm: AsyncOpenAI
    demo_user: UserContext

    async def close(self) -> None:
        await self.embedder.close()
        await self.conversation.close()
        await self.bm25.close()
        await self.llm.close()
        await self.rewriter.client.close()
        await self.engine.dispose()


async def build_container(settings: Settings) -> Container:
    engine, session_factory = create_engine_and_session(settings.database_url)
    embedder = EmbeddingService(
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        model=settings.embedding_model,
        dim=settings.embedding_dim,
        redis_url=settings.redis_url,
    )
    vector_store = MilvusVectorStore(
        uri=settings.milvus_uri,
        collection=settings.milvus_collection,
        dim=settings.embedding_dim,
    )
    bm25 = BM25Index(settings.elasticsearch_url, settings.elasticsearch_index)
    await bm25.ensure_index()
    permission = PermissionService()
    reranker = Reranker(settings.rerank_url)
    rewriter = QueryRewriter(settings.openai_base_url, settings.openai_api_key, settings.chat_model)
    retriever = HybridRetriever(
        embedder=embedder,
        vector_store=vector_store,
        bm25=bm25,
        reranker=reranker,
        permission=permission,
        rrf_k=settings.rrf_k,
        vector_top_k=settings.vector_top_k,
        bm25_top_k=settings.bm25_top_k,
        rerank_top_n=settings.rerank_top_n,
    )
    parser = DocumentParser()
    chunker = StructureAwareChunker()
    knowledge = KnowledgeService(
        session_factory=session_factory,
        parser=parser,
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        bm25=bm25,
    )
    llm = AsyncOpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
    tool_client = InternalApiClient(
        settings.internal_api_base_url,
        settings.internal_api_token,
        settings.tool_timeout_seconds,
    )
    tool_registry = build_default_registry(tool_client)
    tool_planner = ToolPlanner(llm, settings.chat_model)
    citation_builder = CitationBuilder()
    agent = KnowledgeAgent(
        AgentDependencies(
            rewriter=rewriter,
            retriever=retriever,
            tool_planner=tool_planner,
            tool_registry=tool_registry,
            llm=llm,
            citation_builder=citation_builder,
            chat_model=settings.chat_model,
            top_k=settings.default_top_k,
        )
    )
    evaluation = EvaluationService(retriever, citation_builder)
    conversation = ConversationService(settings.redis_url)
    return Container(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        conversation=conversation,
        permission=permission,
        embedder=embedder,
        vector_store=vector_store,
        bm25=bm25,
        reranker=reranker,
        rewriter=rewriter,
        retriever=retriever,
        knowledge=knowledge,
        evaluation=evaluation,
        agent=agent,
        llm=llm,
        demo_user=UserContext(user_id="demo-user", tenant_id="tenant_demo", roles=["admin"]),
    )
