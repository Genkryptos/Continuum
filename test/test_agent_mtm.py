import sys
import types
import unittest

psycopg2_stub = types.SimpleNamespace()
psycopg2_extras_stub = types.SimpleNamespace(RealDictCursor=object, Json=lambda value: value)
sys.modules.setdefault("psycopg2", psycopg2_stub)
sys.modules.setdefault("psycopg2.extras", psycopg2_extras_stub)
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))

from agent.AgentMTM import AgentMTM
from helper.web_search import WebSearchResult
from memory.stm.ConversationSTM import STMConfig, Message


class FakeLLM:
    def __init__(self):
        self.calls = []
        self.model = "fake-model"

    def call_model(self, messages):
        self.calls.append(messages)
        return {
            "response": "ack",
            "success": True,
            "latency": 0.1,
            "tokens_used": {"total": 0},
        }


class FailingLLM(FakeLLM):
    def call_model(self, messages):
        self.calls.append(messages)
        return {"response": None, "success": False, "error": "fail"}


class FakeMTMRetriever:
    def __init__(self, memories=None):
        self.memories = memories or [
            {
                "scope": "episode",
                "source": "test",
                "content": "previous context",
            }
        ]
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.memories


class StubWebSearchService:
    def __init__(self):
        self.calls = []

    def search(self, query, max_results=None):
        self.calls.append({"query": query, "max_results": max_results})
        return [
            WebSearchResult(
                title="Result one",
                url="http://example.com",
                snippet="snippet",
            )
        ]

    def format_results(self, results):
        return "\n".join(f"{item.title} ({item.url})" for item in results)


class TestAgentMTM(unittest.TestCase):
    def test_handle_user_input_uses_mtm_context_and_llm(self):
        llm = FakeLLM()
        retriever = FakeMTMRetriever()
        stm_config = STMConfig(max_tokens=256, reserved_for_response=64, max_messages=5)

        agent = AgentMTM(
            llm=llm,
            mtm_retriever=retriever,
            stm_config=stm_config,
            model_name="gpt-4o",
        )

        result = agent.handle_user_input(
            user_input="hello world",
            user_id="user-1",
            agent_id="agent-1",
            session_key="session-1",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["response"], "ack")
        self.assertEqual(len(result["mtm_memories"]), 1)
        self.assertGreater(len(llm.calls), 0)
        system_messages = [m for m in llm.calls[0] if m["role"] == "system"]
        self.assertTrue(
            any("previous context" in m["content"] for m in system_messages),
            "Expected retrieved context to be included in system messages",
        )

    def test_failure_rolls_back_user_message(self):
        llm = FailingLLM()
        retriever = FakeMTMRetriever()
        agent = AgentMTM(
            llm=llm,
            mtm_retriever=retriever,
            stm_config=STMConfig(max_tokens=256, reserved_for_response=64),
            model_name="gpt-4o",
        )

        result = agent.handle_user_input(
            user_input="will fail",
            user_id="user-1",
            agent_id="agent-1",
            session_key="session-1",
        )

        self.assertFalse(result["success"])
        self.assertTrue(result.get("rolled_back_user_message"))
        self.assertEqual(agent.stm.get_messages(), [])

    def test_naive_summarizer_truncates_on_token_budget(self):
        agent = AgentMTM(llm=FakeLLM(), mtm_retriever=FakeMTMRetriever())
        long_content = " ".join(["beta"] * 60)
        message = Message(
            role="user",
            content=long_content,
            tokens=agent.stm_tokenizer(long_content),
        )
        budget = agent.stm_tokenizer("beta beta beta")

        summary = agent._naive_summarizer([message], budget)

        self.assertLessEqual(agent.stm_tokenizer(summary), budget)
        self.assertTrue(summary)

    def test_web_search_results_added_when_enabled(self):
        llm = FakeLLM()
        retriever = FakeMTMRetriever(memories=[])
        web_search = StubWebSearchService()
        stm_config = STMConfig(max_tokens=256, reserved_for_response=64, max_messages=5)

        agent = AgentMTM(
            llm=llm,
            mtm_retriever=retriever,
            stm_config=stm_config,
            model_name="gpt-4o",
            web_search_service=web_search,
            auto_web_search=True,
            web_search_max_results=2,
        )

        result = agent.handle_user_input(
            user_input="latest news about space",
            user_id="user-1",
            agent_id="agent-1",
            session_key="session-1",
        )

        self.assertTrue(result["success"])
        self.assertEqual(len(web_search.calls), 1)
        self.assertEqual(web_search.calls[0]["query"], "latest news about space")
        self.assertEqual(len(result["web_results"]), 1)

        system_messages = [m for m in llm.calls[0] if m["role"] == "system"]
        self.assertTrue(
            any("Live web search results" in m["content"] for m in system_messages),
            "Web context should be injected into the prompt when enabled",
        )

    def test_time_prompt_injected(self):
        llm = FakeLLM()
        retriever = FakeMTMRetriever(memories=[])
        agent = AgentMTM(
            llm=llm,
            mtm_retriever=retriever,
            stm_config=STMConfig(max_tokens=256, reserved_for_response=64),
            model_name="gpt-4o",
        )

        agent.handle_user_input(
            user_input="what day is today?",
            user_id="u",
            agent_id="a",
            session_key="s",
        )

        system_messages = [m for m in llm.calls[0] if m["role"] == "system"]
        self.assertTrue(
            any("Current datetime" in m["content"] for m in system_messages),
            "System prompt should include current datetime grounding",
        )

    def test_capabilities_prompt_added_when_web_search_enabled(self):
        llm = FakeLLM()
        retriever = FakeMTMRetriever(memories=[])
        web_search = StubWebSearchService()
        agent = AgentMTM(
            llm=llm,
            mtm_retriever=retriever,
            web_search_service=web_search,
            auto_web_search=True,
            stm_config=STMConfig(max_tokens=256, reserved_for_response=64),
            model_name="gpt-4o",
        )

        agent.handle_user_input(
            user_input="check something",
            user_id="u",
            agent_id="a",
            session_key="s",
        )

        system_messages = [m for m in llm.calls[0] if m["role"] == "system"]
        self.assertTrue(
            any("access to live web search results" in m["content"] for m in system_messages),
            "Capability hint should be present when web search is configured",
        )


if __name__ == "__main__":
    unittest.main()
