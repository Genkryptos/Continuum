import unittest

from agent.AgentSTM import AgentSTM
from memory.stm.ConversationSTM import STMConfig, Message


class _FakeLLM:
    def __init__(self, response: str = "ok", success: bool = True):
        self.response = response
        self.success = success
        self.calls = 0
        self.model = "fake-llm"

    def call_model(self, _messages):
        self.calls += 1
        if not self.success:
            return {"success": False, "error": "stopped"}
        return {
            "success": True,
            "response": f"{self.response}-{self.calls}",
            "tokens_used": {"total": 0},
            "latency": 0.0,
        }


class TestAgentSTM(unittest.TestCase):
    def test_compression_path_creates_summary_and_discards_old_messages(self):
        tokenizer = lambda text: len(text)
        config = STMConfig(
            max_tokens=120,
            max_messages=50,
            compress_threshold_ratio=0.4,
            max_summary_tokens=50,
            reserved_for_response=0,
            name="compression-test",
        )
        agent = AgentSTM(
            model_name="dummy",
            llm=_FakeLLM(response="assistant"),
            tokenizer=tokenizer,
            config=config,
        )

        for idx in range(6):
            agent.handle_user_input(f"user-message-{idx}")

        messages = agent.stm.get_messages()
        summary_messages = [m for m in messages if m.meta.get("summary")]

        self.assertTrue(summary_messages, "Expected a summary message after compression")
        self.assertFalse(
            any(m.content == "user-message-0" for m in messages),
            "Original early messages should be replaced by the summary",
        )

    def test_negative_path_only_records_user_message_on_failure(self):
        tokenizer = lambda text: len(text)
        agent = AgentSTM(
            model_name="dummy",
            llm=_FakeLLM(success=False),
            tokenizer=tokenizer,
            config=STMConfig(max_tokens=50, reserved_for_response=0),
        )

        result = agent.handle_user_input("hello failure")

        messages = agent.stm.get_messages()
        self.assertFalse(result["success"])
        self.assertEqual(len(messages), 0)
        self.assertTrue(result.get("rolled_back_user_message"))

    def test_naive_summarizer_respects_budget_with_truncation(self):
        agent = AgentSTM(
            model_name="gpt-3.5-turbo",  # model choice ensures tokenizer uses GPT rules
            llm=_FakeLLM(),
            config=STMConfig(max_tokens=200, reserved_for_response=0),
        )

        long_content = " ".join(["alpha"] * 50)
        message = Message(role="user", content=long_content, tokens=agent.tokenizer(long_content))
        budget = agent.tokenizer("alpha alpha alpha")

        summary = agent._naive_summarizer([message], budget)

        self.assertLessEqual(agent.tokenizer(summary), budget)
        self.assertTrue(summary)


if __name__ == "__main__":
    unittest.main()
