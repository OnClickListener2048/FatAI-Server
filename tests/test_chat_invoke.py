"""Unit tests for the <invoke name="..."> markup recovery in chat tool calling.

deepseek-chat sometimes emits its official-app web-search invocation as plain text
instead of structured tool_calls; _extract_invoke_calls turns that markup back into
a real tool call so the round executes search and continues normally.
"""

import unittest

from app.services.chat import _extract_invoke_calls, _strip_invoke_markup

ALLOWED = frozenset({"web_search", "weather"})


class ExtractInvokeCallsTest(unittest.TestCase):
    def test_single_invoke_with_parameter(self) -> None:
        text = (
            "搜索结果还没有具体数据。让我尝试更针对性的搜索。"
            '<invoke name="web_search"><parameter name="query">蒙古 出口额 煤炭 铜精矿 占比 2023</parameter></invoke>'
        )
        calls = _extract_invoke_calls(text, ALLOWED)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "web_search")
        self.assertEqual(calls[0]["args"], {"query": "蒙古 出口额 煤炭 铜精矿 占比 2023"})
        self.assertEqual(calls[0]["id"], "invoke-0")
        self.assertEqual(calls[0]["type"], "function")

    def test_multiple_invokes_get_sequential_ids(self) -> None:
        text = (
            '<invoke name="web_search"><parameter name="query">a</parameter></invoke>'
            '<invoke name="weather"><parameter name="location">北京</parameter></invoke>'
        )
        calls = _extract_invoke_calls(text, ALLOWED)
        self.assertEqual([call["id"] for call in calls], ["invoke-0", "invoke-1"])
        self.assertEqual(calls[1]["args"], {"location": "北京"})

    def test_unknown_tool_name_is_rejected(self) -> None:
        text = '<invoke name="exec_command"><parameter name="cmd">rm -rf /</parameter></invoke>'
        self.assertEqual(_extract_invoke_calls(text, ALLOWED), [])

    def test_parameter_without_value_keeps_blank_string(self) -> None:
        text = '<invoke name="web_search"><parameter name="query"></parameter></invoke>'
        calls = _extract_invoke_calls(text, ALLOWED)
        self.assertEqual(calls[0]["args"], {"query": ""})

    def test_xml_entities_are_unescaped(self) -> None:
        text = '<invoke name="web_search"><parameter name="query">a &amp; b &lt; c</parameter></invoke>'
        calls = _extract_invoke_calls(text, ALLOWED)
        self.assertEqual(calls[0]["args"], {"query": "a & b < c"})

    def test_markup_stripped_but_narration_kept(self) -> None:
        text = "先搜索一下。<invoke name=\"web_search\"><parameter name=\"query\">x</parameter></invoke>"
        self.assertEqual(_strip_invoke_markup(text), "先搜索一下。")

    def test_no_markup_returns_empty(self) -> None:
        self.assertEqual(_extract_invoke_calls("一个普通的回答", ALLOWED), [])
        self.assertEqual(_extract_invoke_calls("", ALLOWED), [])


if __name__ == "__main__":
    unittest.main()
