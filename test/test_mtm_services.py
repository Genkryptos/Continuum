import sys
import types
import unittest
from unittest.mock import MagicMock, call

# Provide lightweight psycopg2 stubs so MTM modules can import without the real dependency
psycopg2_stub = types.SimpleNamespace(connect=MagicMock())
psycopg2_extras_stub = types.SimpleNamespace(RealDictCursor=object, Json=lambda value: value)
sys.modules.setdefault("psycopg2", psycopg2_stub)
sys.modules.setdefault("psycopg2.extras", psycopg2_extras_stub)
sys.modules.setdefault("openai", types.SimpleNamespace(OpenAI=object))
sys.modules.setdefault("dotenv", types.SimpleNamespace(load_dotenv=lambda: None))


class DummySentenceTransformer:
    def __init__(self, model_name=None):
        self.model_name = model_name
        self.encode_calls = []

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, text, **kwargs):
        self.encode_calls.append({"text": text, **kwargs})
        return [[0.1, 0.2, 0.3] for _ in text]


sys.modules.setdefault(
    "sentence_transformers", types.SimpleNamespace(SentenceTransformer=DummySentenceTransformer)
)

from memory.mtm.mtmCallbacks import MTMCallbacks
from memory.mtm.service.cloudEmbeddingService import OpenAIEmbeddingService
from memory.mtm.service.embeddingService import EmbeddingError
from memory.mtm.service.localEmbeddingService import LocalEmbeddingService
from memory.mtm.service.llmSummarizerService import LLMSummarizerService
from memory.stm.ConversationSTM import Importance, Message


class DummyEmbedding:
    def __init__(self, index, embedding):
        self.index = index
        self.embedding = embedding


class DummyEmbeddingsClient:
    def __init__(self, data):
        self.data = data
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(data=self.data)


class MultiResponseEmbeddingsClient(DummyEmbeddingsClient):
    def __init__(self, data_batches):
        super().__init__(data_batches)
        self._call_index = 0

    def create(self, **kwargs):
        self.calls.append(kwargs)
        batch = self.data[self._call_index]
        self._call_index += 1
        return types.SimpleNamespace(data=batch)


class DummyOpenAIClient:
    def __init__(self, embeddings_data):
        self.embeddings = DummyEmbeddingsClient(embeddings_data)


class DummyChatCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class DummyChat:
    def __init__(self, response):
        self.completions = DummyChatCompletions(response)


class DummyOpenAIChatClient:
    def __init__(self, response):
        self.chat = DummyChat(response)


class TestOpenAIEmbeddingService(unittest.TestCase):
    def test_embed_batch_sorts_and_sets_dimension(self):
        embeddings_data = [
            DummyEmbedding(index=1, embedding=[0.5, 0.6]),
            DummyEmbedding(index=0, embedding=[1.0, 2.0, 3.0]),
        ]
        client = DummyOpenAIClient(embeddings_data)
        service = OpenAIEmbeddingService(model_name="demo-model", client=client)

        vectors = service.embed_batch(["first", "second"])

        self.assertEqual(vectors, [[1.0, 2.0, 3.0], [0.5, 0.6]])
        self.assertEqual(service.embedding_dim, 3)
        self.assertEqual(client.embeddings.calls[0]["model"], "demo-model")
        self.assertEqual(client.embeddings.calls[0]["input"], ["first", "second"])

    def test_embed_text_routes_through_batch(self):
        client = DummyOpenAIClient([DummyEmbedding(index=0, embedding=[0.1, 0.2])])
        service = OpenAIEmbeddingService(client=client)
        service.embed_batch = MagicMock(return_value=[[0.1, 0.2]])

        vector = service.embed_text("hello")

        service.embed_batch.assert_called_once_with(["hello"])
        self.assertEqual(vector, [0.1, 0.2])

    def test_embed_batch_wraps_failures(self):
        class FailingEmbeddingsClient(DummyEmbeddingsClient):
            def create(self, **kwargs):
                raise ValueError("boom")

        client = DummyOpenAIClient([])
        client.embeddings = FailingEmbeddingsClient([])

        service = OpenAIEmbeddingService(model_name="demo-model", client=client)

        with self.assertRaises(EmbeddingError) as ctx:
            service.embed_batch(["first"])

        self.assertIn("demo-model", str(ctx.exception))
        self.assertIn("boom", str(ctx.exception))

    def test_embed_batch_chunks_inputs(self):
        responses = [
            [
                DummyEmbedding(index=1, embedding=[0.2]),
                DummyEmbedding(index=0, embedding=[0.1]),
            ],
            [DummyEmbedding(index=0, embedding=[0.3])],
        ]
        client = DummyOpenAIClient([])
        client.embeddings = MultiResponseEmbeddingsClient(responses)
        service = OpenAIEmbeddingService(model_name="demo-model", client=client)

        vectors = service.embed_batch(["a", "b", "c"], batch_size=2)

        self.assertEqual(vectors, [[0.1], [0.2], [0.3]])
        self.assertEqual(len(client.embeddings.calls), 2)
        self.assertEqual(client.embeddings.calls[0]["input"], ["a", "b"])
        self.assertEqual(client.embeddings.calls[0]["model"], "demo-model")
        self.assertEqual(client.embeddings.calls[1]["input"], ["c"])
        self.assertEqual(service.embedding_dim, 1)


class TestLLMSummarizerService(unittest.TestCase):
    def test_summarize_message_builds_prompt(self):
        response = types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="summary"))]
        )
        client = DummyOpenAIChatClient(response)
        service = LLMSummarizerService(client=client, model_name="summary-model")

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]

        result = service.summarize_message(messages, max_token=50, context="context info")

        self.assertEqual(result, "summary")
        call_kwargs = client.chat.completions.calls[0]
        self.assertEqual(call_kwargs["model"], "summary-model")
        self.assertEqual(call_kwargs["max_tokens"], 50)
        self.assertEqual(call_kwargs["temperature"], 0.3)
        self.assertEqual(call_kwargs["messages"][0]["role"], "system")
        self.assertTrue("context info" in call_kwargs["messages"][1]["content"])


class TestLocalEmbeddingService(unittest.TestCase):
    def test_embed_batch_wraps_failures(self):
        class FailingModel(DummySentenceTransformer):
            def encode(self, text, **kwargs):
                raise RuntimeError("nope")

        service = LocalEmbeddingService(model_name="demo", model=FailingModel("demo"))

        with self.assertRaises(EmbeddingError) as ctx:
            service.embed_batch(["hi"])

        self.assertIn("demo", str(ctx.exception))
        self.assertIn("nope", str(ctx.exception))

    def test_init_wraps_model_load_failures(self):
        from memory.mtm.service import localEmbeddingService as les

        original_model = les.SentenceTransformer

        class FailingLoader:
            def __init__(self, *_, **__):
                raise RuntimeError("bad model")

        try:
            les.SentenceTransformer = FailingLoader

            with self.assertRaises(EmbeddingError) as ctx:
                LocalEmbeddingService(model_name="missing-model")

            self.assertIn("missing-model", str(ctx.exception))
            self.assertIn("bad model", str(ctx.exception))
        finally:
            les.SentenceTransformer = original_model


class TestMTMCallbacks(unittest.TestCase):
    def test_on_compress_persists_summary_and_messages(self):
        repo = MagicMock()
        repo.create_or_get_session.return_value = 42
        embedding_service = MagicMock()
        embedding_service.embed_batch.return_value = [
            [0.0, 0.1],
            [0.2, 0.3],
            [0.4, 0.5],
        ]
        embedding_service.model_name = "mock-model"

        callbacks = MTMCallbacks(
            mtm_repo=repo,
            embedding_service=embedding_service,
            user_id="user-1",
            agent_id="agent-1",
            session_key="session-1",
        )

        summary_message = Message(
            role="system", content="summary", tokens=5, importance=Importance.HIGH
        )
        old_messages = [
            Message(role="user", content="hello", tokens=2, importance=Importance.NORMAL),
            Message(role="assistant", content="hi", tokens=3, importance=Importance.CRITICAL),
        ]

        callbacks.on_compress(old_messages, summary_message)

        repo.create_or_get_session.assert_called_once_with("user-1", "agent-1", "session-1")
        embedding_service.embed_batch.assert_called_once_with(["summary", "hello", "hi"])

        repo.add_memory_batch.assert_called_once()
        stored_batch = repo.add_memory_batch.call_args[0][0]
        self.assertEqual(len(stored_batch), 3)

        summary_entry = stored_batch[0]
        group_id = summary_entry["tags"].get("summary_group_id")

        self.assertEqual(summary_entry["session_id"], 42)
        self.assertEqual(summary_entry["scope"], "stm_summary")
        self.assertEqual(summary_entry["source"], "stm_compress")
        self.assertEqual(summary_entry["content"], "summary")
        self.assertEqual(summary_entry["importance"], Importance.HIGH.value)
        self.assertEqual(
            summary_entry["tags"],
            {
                "type": "conversation_summary",
                "num_messages_compressed": 2,
                "original_importance": "HIGH",
                "summary_group_id": group_id,
            },
        )
        self.assertEqual(summary_entry["embedding"], [0.0, 0.1])
        self.assertEqual(summary_entry["embedding_model"], "mock-model")
        self.assertIsNone(summary_entry["expires_at"])

        first_msg = stored_batch[1]
        self.assertEqual(first_msg["tags"].get("summary_group_id"), group_id)
        self.assertEqual(
            first_msg["tags"],
            {
                "type": "episode",
                "role": "user",
                "original_importance": "NORMAL",
                "summary_group_id": group_id,
            },
        )

        second_msg = stored_batch[2]
        self.assertEqual(second_msg["tags"].get("summary_group_id"), group_id)

    def test_on_evict_persists_user_and_assistant_messages(self):
        repo = MagicMock()
        repo.create_or_get_session.return_value = 77
        embedding_service = MagicMock()
        embedding_service.embed_text.return_value = [9.9, 8.8]

        callbacks = MTMCallbacks(
            mtm_repo=repo,
            embedding_service=embedding_service,
            user_id="user-evict",
            agent_id="agent-evict",
            session_key="session-evict",
        )

        evicted = Message(
            role="user",
            content="evicted message",
            tokens=4,
            importance=Importance.HIGH,
        )

        callbacks.on_evict(evicted)

        repo.create_or_get_session.assert_called_once_with(
            "user-evict", "agent-evict", "session-evict"
        )
        embedding_service.embed_text.assert_called_once_with("evicted message")
        repo.add_memory.assert_called_once_with(
            session_id=77,
            scope="episode",
            source="stm_evict",
            content="evicted message",
            importance=Importance.HIGH.value,
            tags={
                "type": "episode",
                "role": "user",
                "original_importance": "HIGH",
            },
            embeddings=[9.9, 8.8],
            expires_at=None,
        )

        # Non conversational messages should be ignored
        repo.add_memory.reset_mock()
        embedding_service.embed_text.reset_mock()
        callbacks.on_evict(
            Message(role="system", content="ignore", tokens=1, importance=Importance.LOW)
        )
        repo.add_memory.assert_not_called()
        embedding_service.embed_text.assert_not_called()

    def test_on_evict_logs_when_session_unavailable(self):
        repo = MagicMock()
        repo.create_or_get_session.return_value = None
        embedding_service = MagicMock()

        callbacks = MTMCallbacks(
            mtm_repo=repo,
            embedding_service=embedding_service,
            user_id="user-evict",
            agent_id="agent-evict",
            session_key="session-evict",
        )

        with self.assertLogs("memory.mtm.mtmCallbacks", level="WARNING") as log_ctx:
            callbacks.on_evict(
                Message(role="user", content="hello", tokens=1, importance=Importance.LOW)
            )

        self.assertIn("Failed to create or fetch MTM session", " ".join(log_ctx.output))
        embedding_service.embed_text.assert_not_called()
        repo.add_memory.assert_not_called()

    def test_on_evict_logs_only_once_when_session_missing(self):
        repo = MagicMock()
        repo.create_or_get_session.return_value = None
        embedding_service = MagicMock()

        callbacks = MTMCallbacks(
            mtm_repo=repo,
            embedding_service=embedding_service,
            user_id="user-evict",
            agent_id="agent-evict",
            session_key="session-evict",
        )

        with self.assertLogs("memory.mtm.mtmCallbacks", level="WARNING") as log_ctx:
            callbacks.on_evict(
                Message(role="user", content="hello", tokens=1, importance=Importance.LOW)
            )
            callbacks.on_evict(
                Message(role="assistant", content="reply", tokens=1, importance=Importance.LOW)
            )

        warnings = [msg for msg in log_ctx.output if "Failed to create or fetch MTM session" in msg]
        self.assertEqual(len(warnings), 1)
        embedding_service.embed_text.assert_not_called()
        repo.add_memory.assert_not_called()

    def test_on_evict_logs_when_insert_fails(self):
        repo = MagicMock()
        repo.create_or_get_session.return_value = 101
        repo.add_memory.return_value = None
        embedding_service = MagicMock()
        embedding_service.embed_text.return_value = [1.0]

        callbacks = MTMCallbacks(
            mtm_repo=repo,
            embedding_service=embedding_service,
            user_id="user-evict",
            agent_id="agent-evict",
            session_key="session-evict",
        )

        with self.assertLogs("memory.mtm.mtmCallbacks", level="WARNING") as log_ctx:
            callbacks.on_evict(
                Message(role="user", content="hello", tokens=1, importance=Importance.LOW)
            )

        self.assertTrue(any("Failed to persist MTM memory" in msg for msg in log_ctx.output))
        embedding_service.embed_text.assert_called_once_with("hello")

    def test_on_evict_logs_and_recovers_from_errors(self):
        repo = MagicMock()
        repo.create_or_get_session.return_value = 101

        def raising_embed(_):
            raise RuntimeError("rate limit")

        embedding_service = MagicMock()
        embedding_service.embed_text.side_effect = raising_embed

        callbacks = MTMCallbacks(
            mtm_repo=repo,
            embedding_service=embedding_service,
            user_id="user-evict",
            agent_id="agent-evict",
            session_key="session-evict",
        )

        with self.assertLogs("memory.mtm.mtmCallbacks", level="ERROR") as log_ctx:
            callbacks.on_evict(
                Message(role="user", content="boom", tokens=1, importance=Importance.LOW)
            )

        self.assertTrue(any("Failed to store evicted message" in msg for msg in log_ctx.output))
        repo.add_memory.assert_not_called()

    def test_prunes_when_configured(self):
        repo = MagicMock()
        repo.create_or_get_session.return_value = 202
        embedding_service = MagicMock()
        embedding_service.embed_text.return_value = [3.14]

        callbacks = MTMCallbacks(
            mtm_repo=repo,
            embedding_service=embedding_service,
            user_id="user-prune",
            agent_id="agent-prune",
            session_key="session-prune",
            max_memories_per_user=10,
        )

        callbacks.on_evict(
            Message(role="user", content="keep track", tokens=1, importance=Importance.LOW)
        )

        repo.prune_per_user_caps.assert_called_once_with(user_id="user-prune", max_per_user=10)

    def test_on_compress_recovers_from_partial_failures(self):
        repo = MagicMock()
        repo.create_or_get_session.return_value = 55

        embedding_service = MagicMock()
        embedding_service.embed_batch.side_effect = RuntimeError("batch down")

        def embed_text(text):
            if text == "fail":
                raise RuntimeError("embedding down")
            return [42]

        embedding_service.embed_text.side_effect = embed_text
        embedding_service.model_name = "mock-model"

        callbacks = MTMCallbacks(
            mtm_repo=repo,
            embedding_service=embedding_service,
            user_id="user-compress",
            agent_id="agent-compress",
            session_key="session-compress",
        )

        old_messages = [
            Message(role="user", content="ok", tokens=1, importance=Importance.NORMAL),
            Message(role="assistant", content="fail", tokens=1, importance=Importance.LOW),
            Message(role="assistant", content="ok2", tokens=1, importance=Importance.LOW),
        ]
        summary = Message(role="system", content="summary", tokens=1, importance=Importance.HIGH)

        with self.assertLogs("memory.mtm.mtmCallbacks", level="ERROR") as log_ctx:
            callbacks.on_compress(old_messages, summary)

        embedding_service.embed_batch.assert_called_once_with(
            ["summary", "ok", "fail", "ok2"]
        )
        self.assertTrue(
            any("Failed to store compressed message" in msg for msg in log_ctx.output)
        )

        # Summary and successful messages should still be persisted
        repo.add_memory_batch.assert_called_once()
        stored_batch = repo.add_memory_batch.call_args[0][0]
        self.assertEqual(len(stored_batch), 3)
        summary_entry = stored_batch[0]
        group_id = summary_entry["tags"].get("summary_group_id")
        self.assertEqual(summary_entry["scope"], "stm_summary")
        self.assertEqual(summary_entry["embedding"], [42])
        self.assertEqual(summary_entry["tags"]["num_messages_compressed"], 3)
        self.assertIsNotNone(group_id)

        ok_entry = stored_batch[1]
        self.assertEqual(ok_entry["content"], "ok")
        self.assertEqual(ok_entry["importance"], Importance.NORMAL.value)
        self.assertEqual(ok_entry["tags"].get("summary_group_id"), group_id)

        ok2_entry = stored_batch[2]
        self.assertEqual(ok2_entry["content"], "ok2")
        self.assertEqual(ok2_entry["tags"]["role"], "assistant")
        self.assertEqual(ok2_entry["tags"].get("summary_group_id"), group_id)


if __name__ == "__main__":
    unittest.main()
