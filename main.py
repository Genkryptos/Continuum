import logging
import os

from dotenv import load_dotenv

from agent.AgentMTM import AgentMTM
from memory.mtm.mtmCallbacks import MTMCallbacks
from memory.mtm.repository.mtmRepository import MTMRepository
from memory.mtm.repository.mtmRetriever import MTMRetriever
from memory.mtm.service.cloudEmbeddingService import OpenAIEmbeddingService
from settings import LLM_MODEL, MTM_MAX_PER_USER


def main() -> None:
    # Load .env (OPEN_AI_KEY, LLM, EMBEDDINGS, etc.)
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # For now, fix user/agent/session identifiers for the CLI demo
    user_id = os.getenv("AGENT_USER_ID", "cli-user")
    agent_id = os.getenv("AGENT_ID", "default-agent")
    session_key = os.getenv("SESSION_KEY", "cli-session")

    # --- MTM infrastructure setup -------------------------------------------
    db_url = os.getenv("MTM_DB_URL")
    if not db_url:
        raise RuntimeError(
            "MTM_DB_URL is not set. Configure it in your .env, e.g.\n"
            "MTM_DB_URL=postgresql://myuser:mypassword@localhost:5433/mydb"
        )

    mtm_repo = MTMRepository(db_url=db_url)
    embedding_service = OpenAIEmbeddingService()  # uses EMBEDDINGS from settings/env internally
    mtm_retriever = MTMRetriever(mtm_repo=mtm_repo, embedding_service=embedding_service)

    mtm_callbacks = MTMCallbacks(
        mtm_repo=mtm_repo,
        embedding_service=embedding_service,
        user_id=user_id,
        agent_id=agent_id,
        session_key=session_key,
        max_memories_per_user=MTM_MAX_PER_USER or None,
    )

    # --- AgentMTM (STM + MTM) ----------------------------------------------
    agent = AgentMTM(
        model_name=LLM_MODEL,     # from env: LLM=gpt-4o-mini etc.
        mtm_retriever=mtm_retriever,
        stm_callbacks=mtm_callbacks,
        # you can override max_context_tokens / answer_fraction / mtm_top_k here if needed
    )

    print(f"AgentMTM (STM + MTM) using model: {LLM_MODEL}")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        result = agent.handle_user_input(
            user_input=user,
            user_id=user_id,
            agent_id=agent_id,
            session_key=session_key,
        )

        if not result["success"]:
            print(f"Error: {result['error']}")
            continue

        print(f"Assistant: {result['response']}\n")

        s_after = result["stm_stats_after"]
        print(
            f"[STM] msgs={s_after['messages']} "
            f"tokens={s_after['tokens']} "
            f"util={s_after['utilization']:.1%}\n"
        )


if __name__ == "__main__":
    main()
