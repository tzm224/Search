import unittest

from search_agent.parser import (
    GoogleSearchParser,
    extract_answer_block,
    extract_prediction_text,
    strip_citations,
    truncate_at_tool_call,
)


class ParserTests(unittest.TestCase):
    def test_extracts_multiple_tool_calls(self) -> None:
        parser = GoogleSearchParser()
        content, tool_calls = parser.extract_tool_calls(
            "thinking\n<google_search>alpha</google_search>\n<google_search>beta</google_search>"
        )

        self.assertEqual(content, "thinking")
        self.assertEqual(
            [call.arguments["query_list"][0] for call in tool_calls],
            ["alpha", "beta"],
        )

    def test_extract_answer_block_returns_inner_text(self) -> None:
        text = "<think>...</think>\n<answer>Paris<cite id=\"S_1\">Evidence</cite></answer>"
        self.assertEqual(
            extract_answer_block(text),
            "Paris<cite id=\"S_1\">Evidence</cite>",
        )

    def test_extract_prediction_text_removes_citations_and_normalizes_space(self) -> None:
        text = "<answer>\nParis  <cite id=\"S_1\">Evidence</cite>\n</answer>"
        self.assertEqual(strip_citations(extract_answer_block(text)), "Paris")
        self.assertEqual(extract_prediction_text(text), "Paris")

    def test_truncate_at_tool_call_keeps_closing_tag(self) -> None:
        text = "<think>x</think><google_search>foo</google_search><answer>bar</answer>"
        self.assertEqual(
            truncate_at_tool_call(text),
            "<think>x</think><google_search>foo</google_search>",
        )


if __name__ == "__main__":
    unittest.main()
