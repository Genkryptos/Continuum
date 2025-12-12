import unittest
from typing import Mapping
from unittest.mock import MagicMock

from memory.stm.ConversationSTM import (
    ConversationSTM,
    Importance,
    Message,
    STMConfig, STMCallbacks,
)
from memory.stm.SharedSTM import SharedSTM
from memory.stm.ThreadSafeSTM import ThreadSafeSTM


def simple_tokenizer(text: str) -> int:
    return len(str(text))


class TestConversationSTM(unittest.TestCase):
    def test_add_messages_and_stats(self):
        config = STMConfig(max_tokens=50, reserved_for_response=0, max_messages=10)
        stm = ConversationSTM(config, simple_tokenizer)

        stm.add_user_message("hello")
        stm.add_assistant_message("world!", importance=Importance.HIGH)

        messages = stm.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0].role, "user")
        self.assertEqual(messages[1].importance, Importance.HIGH)

        stats = stm.stats()
        self.assertEqual(stats["messages"], 2)
        self.assertEqual(stats["tokens"], len("hello") + len("world!"))
        self.assertAlmostEqual(stats["utilization"], (len("hello") + len("world!")) / 50)

    def test_eviction_prefers_less_important_messages(self):
        config = STMConfig(max_tokens=15, reserved_for_response=0, max_messages=10)
        stm = ConversationSTM(config, simple_tokenizer)

        stm.add_system_message("hi", importance=Importance.HIGH)
        stm.add_user_message("norm", importance=Importance.NORMAL)

        stm.add_assistant_message("new message")

        roles = [m.role for m in stm.get_messages()]
        contents = [m.content for m in stm.get_messages()]

        self.assertIn("system", roles)
        self.assertNotIn("norm", contents)
        self.assertIn("new message", contents)

    def test_maybe_compress_adds_summary_when_threshold_met(self):
        config = STMConfig(
            max_tokens=100,
            reserved_for_response=0,
            max_messages=50,
            compress_threshold_ratio=0.5,
        )
        stm = ConversationSTM(config, simple_tokenizer)

        # Create enough messages to trigger compression
        for i in range(10):
            stm.add_user_message(f"msg-{i}")

        summary_text = "summary"
        summarizer = MagicMock(return_value=summary_text)

        stm.maybe_compress(summarizer_fn=summarizer, min_messages_to_compress=4)

        summarizer.assert_called_once()
        args, _ = summarizer.call_args
        self.assertIsInstance(args[0], list)
        self.assertIsInstance(args[1], int)
        self.assertGreater(args[1], 0)
        messages = stm.get_messages()

        self.assertEqual(messages[0].role, "system")
        self.assertTrue(messages[0].meta.get("summary"))
        self.assertEqual(len(messages), 6)  # summary + 5 recent messages

        expected_tokens = len(summary_text) + sum(len(f"msg-{i}") for i in range(5, 10))
        self.assertEqual(sum(m.tokens for m in messages), expected_tokens)

    def test_summary_is_retained_when_eviction_needed(self):
        config = STMConfig(
            max_tokens=40,
            reserved_for_response=0,
            max_messages=10,
            compress_threshold_ratio=0.5,
        )
        stm = ConversationSTM(config, simple_tokenizer)

        for i in range(8):
            stm.add_user_message(f"msg-{i}")

        summarizer = MagicMock(return_value="summary-text")
        stm.maybe_compress(summarizer_fn=summarizer, min_messages_to_compress=4)

        stm.add_assistant_message("incoming-long-message")

        messages = stm.get_messages()
        self.assertTrue(any(m.meta.get("summary") for m in messages))
        self.assertTrue(any(m.content == "incoming-long-message" for m in messages))
        self.assertLessEqual(sum(m.tokens for m in messages), config.max_tokens)

    def test_compress_fraction_applies_when_configured(self):
        config = STMConfig(
            max_tokens=60,
            reserved_for_response=0,
            max_messages=50,
            compress_threshold_ratio=0.4,
            compress_fraction=0.8,
        )
        stm = ConversationSTM(config, simple_tokenizer)

        for i in range(10):
            stm.add_user_message(f"msg-{i}")

        summarizer = MagicMock(return_value="summary")

        stm.maybe_compress(summarizer_fn=summarizer, min_messages_to_compress=4)

        messages = stm.get_messages()
        self.assertTrue(messages[0].meta.get("summary"))
        # compress_fraction=0.8 with 10 messages should summarize 8 and keep 2
        self.assertEqual(len(messages), 3)
        self.assertListEqual([m.content for m in messages[1:]], ["msg-8", "msg-9"])

    def test_compression_skipped_when_summary_too_large(self):
        config = STMConfig(
            max_tokens=10,
            reserved_for_response=0,
            max_messages=10,
            compress_threshold_ratio=0.3,
        )
        stm = ConversationSTM(config, simple_tokenizer)

        for i in range(4):
            stm.add_user_message(f"m{i}")

        summarizer = MagicMock(return_value="x" * 20)
        stm.maybe_compress(summarizer_fn=summarizer, min_messages_to_compress=2)

        messages = stm.get_messages()
        self.assertEqual(len(messages), 4)
        self.assertFalse(any(m.meta.get("summary") for m in messages))

    def test_critical_messages_retained_when_capacity_reached(self):
        config = STMConfig(max_tokens=12, reserved_for_response=0, max_messages=10)
        stm = ConversationSTM(config, simple_tokenizer)

        stm.add_user_message("abcd", importance=Importance.NORMAL)
        stm.add_user_message("efgh", importance=Importance.LOW)
        stm.add_assistant_message("crit", importance=Importance.CRITICAL)

        stm.add_system_message("newmsg")

        messages = stm.get_messages()

        self.assertTrue(any(m.content == "crit" and m.importance == Importance.CRITICAL for m in messages))
        self.assertIn("newmsg", [m.content for m in messages])
        self.assertLessEqual(sum(m.tokens for m in messages), config.max_tokens)

    def test_reserved_for_response_reduces_effective_budget(self):
        config = STMConfig(max_tokens=20, reserved_for_response=10, max_messages=10)
        stm = ConversationSTM(config, simple_tokenizer)

        stm.add_user_message("12345")  # 5
        stm.add_user_message("67890")  # 5 -> total 10

        # effective_budget = 10; adding more should cause eviction
        stm.add_assistant_message("ABCDEFGHIJ")  # 10

        total_tokens = sum(m.tokens for m in stm.get_messages())
        self.assertLessEqual(total_tokens, config.max_tokens - config.reserved_for_response)

    def test_generate_summary_retries_until_fit(self):
        config = STMConfig(
            max_tokens=30,
            reserved_for_response=0,
            max_messages=20,
            compress_threshold_ratio=0.3,
            max_summary_tokens=20,
        )
        stm = ConversationSTM(config, simple_tokenizer)

        for i in range(6):
            stm.add_user_message(f"msg-{i}")

        call_budgets = []

        def summarizer(msgs, budget):
            call_budgets.append(budget)
            return "X" * 50 if len(call_budgets) == 1 else "Y" * 10

        stm.maybe_compress(summarizer_fn=summarizer, min_messages_to_compress=4)

        self.assertGreaterEqual(len(call_budgets), 2)
        messages = stm.get_messages()

        self.assertTrue(messages[0].meta.get("summary"))
        self.assertIn("Y" * 10, messages[0].content)

    def test_compress_fraction_scales_with_utilization(self):
        config = STMConfig(
            max_tokens=30,
            reserved_for_response=0,
            max_messages=20,
            compress_threshold_ratio=0.3,
            compress_fraction=0.2,
        )
        stm = ConversationSTM(config, simple_tokenizer)

        for i in range(6):
            stm.add_user_message(f"msg-{i}")

        summarizer = MagicMock(return_value="summary")

        stm.maybe_compress(summarizer_fn=summarizer, min_messages_to_compress=4)

        messages = stm.get_messages()
        self.assertTrue(messages[0].meta.get("summary"))
        # utilization at 1.0 should escalate compression to summarize the majority
        self.assertLessEqual(len(messages), 3)

    def test_generate_summary_fails_after_all_attempts(self):
        config = STMConfig(
            max_tokens=30,
            reserved_for_response=0,
            max_messages=20,
            compress_threshold_ratio=0.3,
            max_summary_tokens=10,
        )
        stm = ConversationSTM(config, simple_tokenizer)

        for i in range(6):
            stm.add_user_message(f"msg-{i}")

        def summarizer(msgs, budget):
            return "x" * 1000

        stm.maybe_compress(summarizer_fn=summarizer, min_messages_to_compress=4)

        messages = stm.get_messages()
        self.assertEqual(len(messages), 6)
        self.assertFalse(any(m.meta.get("summary") for m in messages))

    def test_enforce_limits_eventually_evicts_summary_when_stuck(self):
        config = STMConfig(
            max_tokens=50,
            reserved_for_response=0,
            max_messages=0,
        )
        stm = ConversationSTM(config, simple_tokenizer)

        summary_msg = Message(
            role="system",
            content="summary",
            tokens=len("summary"),
            importance=Importance.HIGH,
            meta={"summary": True},
        )

        # Use reset to simulate an existing summary-only STM while forcing a
        # max_messages breach that previously could not be resolved.
        stm._reset_messages([summary_msg])

        self.assertEqual(len(stm.get_messages()), 0)
        self.assertEqual(sum(m.tokens for m in stm.get_messages()), 0)

    def test_message_meta_is_deeply_immutable(self):
        meta = {
            "level1": {
                "level2": ["a", "b"],
                "set_val": {"x", "y"},
            }
        }
        msg = Message(role="user", content="x", tokens=1, meta=meta)

        self.assertIsInstance(msg.meta, Mapping)
        self.assertIsInstance(msg.meta["level1"]["level2"], tuple)

        with self.assertRaises(TypeError):
            msg.meta["level1"]["level2"][0] = "c"

        with self.assertRaises(TypeError):
            msg.meta["new_key"] = 123

    def test_message_with_meta_creates_new_frozen_copy(self):
        meta = {"answered": False}
        msg = Message(role="user", content="question", tokens=3, meta=meta)

        updated = msg.with_meta({"answered": True, "notes": {"foo": "bar"}})

        self.assertNotEqual(id(msg), id(updated))
        self.assertFalse(msg.meta.get("answered"))
        self.assertTrue(updated.meta.get("answered"))
        self.assertEqual(msg.timestamp, updated.timestamp)
        self.assertEqual(msg.tokens, updated.tokens)

        with self.assertRaises(TypeError):
            updated.meta["notes"]["foo"] = "baz"

    def test_callbacks_triggered_on_compress_and_evict(self):
        compress_callback = MagicMock()
        evict_callback = MagicMock()
        callbacks = STMCallbacks(on_compress=compress_callback, on_evict=evict_callback)

        config = STMConfig(
            max_tokens=25,
            reserved_for_response=0,
            max_messages=10,
            compress_threshold_ratio=0.4,
        )
        stm = ConversationSTM(config, simple_tokenizer, callbacks=callbacks)

        for i in range(5):
            stm.add_user_message(f"msg-{i}")

        summarizer = MagicMock(return_value="summary")
        stm.maybe_compress(summarizer_fn=summarizer, min_messages_to_compress=4)

        compress_callback.assert_called_once()
        args, _ = compress_callback.call_args
        old_msgs, summary_msg = args
        self.assertIsInstance(old_msgs, list)
        self.assertTrue(summary_msg.meta.get("summary"))

        stm.add_user_message("long-message-that-forces-eviction")

        evict_callback.assert_called()
        evicted_msg = evict_callback.call_args[0][0]
        self.assertIsInstance(evicted_msg, Message)

    def test_on_evict_exception_does_not_block_eviction(self):
        evict_callback = MagicMock(side_effect=RuntimeError("evict failure"))
        callbacks = STMCallbacks(on_evict=evict_callback)

        config = STMConfig(max_tokens=10, reserved_for_response=0, max_messages=2)
        stm = ConversationSTM(config, simple_tokenizer, callbacks=callbacks)

        stm.add_user_message("12345")
        stm.add_assistant_message("67890")

        # This message should trigger eviction despite callback errors
        stm.add_user_message("abc")

        self.assertEqual(len(stm.get_messages()), 2)
        self.assertLessEqual(sum(m.tokens for m in stm.get_messages()), config.max_tokens)

    def test_on_compress_exception_does_not_block_compression(self):
        compress_callback = MagicMock(side_effect=RuntimeError("compress failure"))
        callbacks = STMCallbacks(on_compress=compress_callback)

        config = STMConfig(
            max_tokens=30,
            reserved_for_response=0,
            max_messages=10,
            compress_threshold_ratio=0.3,
        )
        stm = ConversationSTM(config, simple_tokenizer, callbacks=callbacks)

        for i in range(6):
            stm.add_user_message(f"msg-{i}")

        summarizer = MagicMock(return_value="summary")

        stm.maybe_compress(summarizer_fn=summarizer, min_messages_to_compress=4)

        self.assertTrue(any(m.meta.get("summary") for m in stm.get_messages()))
        self.assertLessEqual(sum(m.tokens for m in stm.get_messages()), config.max_tokens)

    def test_threadsafe_delegates_to_wrapped_stm(self):
        base_stm = MagicMock()
        safe_stm = ThreadSafeSTM(base_stm)

        safe_stm.add_user_message("hello")
        safe_stm.add_assistant_message("hi")
        safe_stm.add_system_message("sys")
        safe_stm.add_tool_message("tool")
        safe_stm.get_messages()
        safe_stm.get_prompt_messages(system_prompt="prompt")
        safe_stm.stats()
        safe_stm.maybe_compress(lambda msgs: "")
        safe_stm.rollback_last_user_message()

        base_stm.add_user_message.assert_called_once_with("hello")
        base_stm.add_assistant_message.assert_called_once_with("hi")
        base_stm.add_system_message.assert_called_once_with("sys")
        base_stm.add_tool_message.assert_called_once_with("tool")
        base_stm.get_messages.assert_called_once()
        base_stm.get_prompt_messages.assert_called_once_with(system_prompt="prompt")
        base_stm.stats.assert_called_once()
        base_stm.maybe_compress.assert_called_once()
        base_stm.rollback_last_user_message.assert_called_once()

    def test_shared_stm_filters_and_formats_prompt(self):
        config = STMConfig(max_tokens=100, reserved_for_response=0)
        conv_stm = ConversationSTM(config, simple_tokenizer)
        shared_stm = SharedSTM(ThreadSafeSTM(conv_stm))

        shared_stm.add_message_for_all("user", "keep", meta={"for": "agent-a"})
        shared_stm.add_message_for_all("assistant", "drop", meta={"for": "agent-b"})

        def filter_fn(message: Message, agent_id: str) -> bool:
            return message.meta.get("for") != agent_id

        prompt = shared_stm.get_view_for_agent("agent-a", filter_fn=filter_fn, system_prompt="sys")

        self.assertEqual(prompt[0], {"role": "system", "content": "sys"})
        self.assertEqual(len(prompt), 2)
        self.assertEqual(prompt[1]["content"], "drop")


if __name__ == "__main__":
    unittest.main()