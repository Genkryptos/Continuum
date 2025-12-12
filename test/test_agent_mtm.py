import sys
import types
import unittest

psycopg2_stub = types.SimpleNamespace()
psycopg2_extras_stub = types.SimpleNamespace(RealDictCursor=object, Json=lambda value: value)
sys.modules.setdefault("psycopg2", psycopg2_stub)
sys.modules.setdefault("psycopg2.extras", psycopg2_extras_stub)
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))

from agent.AgentMTM import AgentMTM
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


if __name__ == "__main__":
    unittest.main()
