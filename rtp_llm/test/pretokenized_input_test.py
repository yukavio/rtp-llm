import unittest
from unittest.mock import MagicMock, patch, sentinel

from rtp_llm.config.exceptions import FtRuntimeException
from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.pipeline.pipeline import Pipeline
from rtp_llm.structure.request_constants import request_id_field_name
from rtp_llm.structure.request_extractor import RequestExtractor


class PretokenizedInputTest(unittest.TestCase):
    def setUp(self):
        self.pipeline = Pipeline.__new__(Pipeline)
        self.pipeline.tokenizer = MagicMock()
        self.pipeline.tokenizer.__len__.return_value = 1000
        self.pipeline._special_tokens = MagicMock()
        self.pipeline.create_generate_config = MagicMock(return_value=GenerateConfig())
        self.pipeline.generate_stream = MagicMock(return_value=sentinel.stream)
        self.metrics = patch("rtp_llm.pipeline.pipeline.kmonitor")
        self.metrics.start()
        self.addCleanup(self.metrics.stop)

    def test_raw_request_preserves_ids_through_pipeline(self):
        ids = [0, 999, 25, 25]
        request, kwargs = RequestExtractor(GenerateConfig()).extract_request(
            {request_id_field_name: 7, "input_ids": ids}
        )
        self.assertFalse(request.batch_infer)
        stream = self.pipeline.pipeline_async(
            request.input_texts[0], request_id=request.request_id, **kwargs
        )
        self.assertIs(stream, sentinel.stream)
        self.pipeline.tokenizer.encode.assert_not_called()
        actual = self.pipeline.generate_stream.call_args.args[1]
        self.assertEqual(actual, ids)
        self.assertIsNot(actual, ids)

    def test_ambiguous_request_is_rejected(self):
        for key, value in (("prompt", "a"), ("text", "a"), ("prompt_batch", ["a"])):
            with self.subTest(key=key), self.assertRaises(FtRuntimeException):
                RequestExtractor(GenerateConfig()).extract_request(
                    {request_id_field_name: 7, "input_ids": [1], key: value}
                )

    def test_invalid_ids_are_rejected_before_backend(self):
        for ids in ([], None, "1", [True], [1.0], [-1], [1000], [[1]]):
            with self.subTest(ids=ids), self.assertRaises(FtRuntimeException):
                self.pipeline.pipeline_async("", input_ids=ids)
        self.pipeline.generate_stream.assert_not_called()

    def test_text_path_still_uses_tokenizer(self):
        self.pipeline.tokenizer.encode.return_value = [12, 34]
        self.pipeline.pipeline_async("hello")
        self.pipeline.tokenizer.encode.assert_called_once_with("hello")
        self.assertEqual(self.pipeline.generate_stream.call_args.args[1], [12, 34])

    def test_text_and_multimodal_inputs_cannot_accompany_ids(self):
        for kwargs in ({"prompt": "hello"}, {"prompt": "", "urls": ["image"]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(FtRuntimeException):
                self.pipeline.pipeline_async(input_ids=[1], **kwargs)
        self.pipeline.generate_stream.assert_not_called()

    def test_automatic_text_parsing_requires_text(self):
        self.pipeline.create_generate_config.return_value.auto_parse_banned_combo = True
        with self.assertRaises(FtRuntimeException):
            self.pipeline.pipeline_async("", input_ids=[1])


if __name__ == "__main__":
    unittest.main()
