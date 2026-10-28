import copy
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from langchain_core.language_models.fake_chat_models import FakeListChatModel
import agent_nodes as nodes
from main import run_interview, main
from utils import extract_text_from_file

GRADE = json.dumps({"연관성": "상", "구체성": "중", "평가_의견": "구체적인 답변"}, ensure_ascii=False)


def state():
    return dict(resume_text="합성 이력서", resume_summary="개발 경험", resume_keywords=["Python"],
                question_strategy={"면접 질문 전략": {"경험": {"예시 질문": ["Q1", "Q2"]},
                                  "협업": {"예시 질문": ["협업 경험은?", "갈등 해결은?"]}}},
                current_question="Q1", current_answer="프로젝트에서 맡은 작업을 설명합니다.",
                current_strategy="경험", question_queue=["Q2"], remaining_topics=["협업"],
                generate_count=0, conversation=[], evaluation=[], next_step="evaluate")


class FlowTests(unittest.TestCase):
    def invoke(self, initial, responses):
        model = FakeListChatModel(responses=responses)
        with patch.object(nodes, "get_llm", return_value=model):
            result = nodes.graph.invoke(initial)
        return result, model

    def test_one_invoke_evaluates_only_one_answer(self):
        initial = state(); before = copy.deepcopy(initial)
        result, _ = self.invoke(initial, [GRADE, "next_question"])
        self.assertEqual(len(result["conversation"]), 1)
        self.assertEqual(len(result["evaluation"]), 1)
        self.assertEqual(result["current_question"], "Q2")
        self.assertEqual(result["current_answer"], "")
        self.assertEqual(initial, before)

    def test_next_topic_prompt_has_previous_answer_and_target(self):
        initial = state(); initial["question_queue"] = []
        model = FakeListChatModel(responses=[GRADE, "next_topic", "프로젝트 경험을 바탕으로 협업 사례를 말씀해 주세요."])
        prompts = []
        original = model._call
        def capture(messages, *args, **kwargs):
            prompts.append("\n".join(str(m.content) for m in messages))
            return original(messages, *args, **kwargs)
        with patch.object(nodes, "get_llm", return_value=model), patch.object(type(model), "_call", side_effect=capture):
            result = nodes.graph.invoke(initial)
        self.assertIn(initial["current_answer"], prompts[-1])
        self.assertIn("협업 경험은?", prompts[-1])
        self.assertIn("답변에 없는 경험이나 성과를 만들지", prompts[-1])
        self.assertEqual(result["current_strategy"], "협업")
        self.assertEqual(result["remaining_topics"], [])
        self.assertEqual(result["question_queue"], ["갈등 해결은?"])

    def test_followup_uses_history(self):
        initial = state()
        prompts = []
        model = FakeListChatModel(responses=[GRADE, "generate", "본인의 구체적인 역할은 무엇인가요?"])
        original = model._call
        def capture(messages, *args, **kwargs):
            prompts.append("\n".join(str(m.content) for m in messages))
            return original(messages, *args, **kwargs)
        with patch.object(nodes, "get_llm", return_value=model), patch.object(type(model), "_call", side_effect=capture):
            result = nodes.graph.invoke(initial)
        self.assertIn(initial["current_answer"], prompts[-1])
        self.assertIn("구체적인 답변", prompts[-1])
        self.assertEqual(result["generate_count"], 1)
        self.assertEqual(result["current_answer"], "")
        self.assertEqual(len(result["conversation"]), 1)

    def test_five_user_answers_are_five_evaluations(self):
        initial = state(); initial["question_queue"] = ["Q2", "Q3", "Q4", "Q5"]
        initial["remaining_topics"] = []
        responses = [GRADE, "next_question"] * 4 + [GRADE]
        answers = iter(["A1", "A2", "A3", "A4", "A5"])
        model = FakeListChatModel(responses=responses)
        with patch.object(nodes, "get_llm", return_value=model):
            result = run_interview(initial, input_fn=lambda _: next(answers), output_fn=lambda _: None)
        self.assertEqual([c["answer"] for c in result["conversation"]], ["A1", "A2", "A3", "A4", "A5"])
        self.assertEqual(len(result["evaluation"]), 5)
        self.assertEqual(result["next_step"], "summarize")

    def test_hard_limit_does_not_call_model(self):
        initial = state(); initial["conversation"] = [{}] * 5
        with patch.object(nodes, "get_llm", side_effect=AssertionError("model should not be called")):
            self.assertEqual(nodes.decide_next_step(initial)["next_step"], "summarize")

    def test_invalid_action_and_missing_queue_fall_back(self):
        result, _ = self.invoke(state(), [GRADE, "invalid action"])
        self.assertEqual(result["current_question"], "Q2")
        initial = state(); initial["question_queue"] = []; initial["remaining_topics"] = []
        result, _ = self.invoke(initial, [GRADE, "next_question"])
        self.assertEqual(result["next_step"], "summarize")

    def test_followup_limit_consumes_queue(self):
        initial = state(); initial["generate_count"] = nodes.MAX_FOLLOWUPS
        result, _ = self.invoke(initial, [GRADE, "generate"])
        self.assertEqual(result["current_question"], "Q2")
        self.assertEqual(result["generate_count"], 0)

    def test_empty_answer_is_not_evaluated(self):
        initial = state(); initial["current_answer"] = "  "
        with patch.object(nodes, "get_llm", side_effect=AssertionError("no model call")):
            with self.assertRaises(ValueError): nodes.graph.invoke(initial)

    def test_invalid_grade_preserves_input_logs(self):
        initial = state()
        with self.assertRaises(ValueError): self.invoke(initial, ['{"연관성":"unknown"}'])
        self.assertEqual(initial["conversation"], [])
        self.assertEqual(initial["evaluation"], [])

    def test_strategy_initialization_accepts_flat_schema(self):
        initial = state()
        model = FakeListChatModel(responses=[json.dumps({"협업": {"예시 질문": ["Q1", "Q2"]}}, ensure_ascii=False)])
        with patch.object(nodes, "get_llm", return_value=model):
            result = nodes.generate_question_strategy(initial)
        self.assertEqual(result["current_strategy"], "협업")
        self.assertEqual(result["remaining_topics"], [])
        self.assertEqual(result["question_queue"], ["Q2"])
        with self.assertRaises(ValueError): nodes.normalize_strategy({"협업": {"예시 질문": "bad"}})

    def test_blank_cli_and_quit_never_invoke_graph(self):
        answers = iter([" ", "/quit"])
        class Never:
            def invoke(self, _): raise AssertionError("no evaluation")
        result = run_interview(state(), graph_runner=Never(), input_fn=lambda _: next(answers), output_fn=lambda _: None)
        self.assertEqual(result["conversation"], [])
        self.assertEqual(result["next_step"], "summarize")

    def test_missing_file_and_key_exit_cleanly(self):
        with redirect_stdout(io.StringIO()), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(main(["missing.pdf"]), 1)
            with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
                self.assertEqual(main([f.name]), 1)

    def test_pdf_and_docx_extraction(self):
        import fitz
        from docx import Document
        with tempfile.TemporaryDirectory() as d:
            pdf = Path(d) / "resume.pdf"; docx = Path(d) / "resume.docx"
            doc = fitz.open(); page = doc.new_page(); page.insert_text((50, 50), "Synthetic resume")
            doc.save(pdf); doc.close()
            word = Document(); word.add_paragraph("Synthetic resume"); word.save(docx)
            self.assertIn("Synthetic resume", extract_text_from_file(str(pdf)))
            self.assertIn("Synthetic resume", extract_text_from_file(str(docx)))

    def test_resume_preparation_to_first_question(self):
        model = FakeListChatModel(responses=["개발 경험 요약", "Python, 협업",
            json.dumps({"면접 질문 전략": {"경력 및 경험": {"예시 질문": ["첫 질문", "둘째 질문"]},
                                        "협업": {"예시 질문": ["협업 질문"]}}}, ensure_ascii=False)])
        with patch.object(nodes, "get_llm", return_value=model), patch.object(nodes, "extract_text_from_file", return_value="합성 이력서"):
            prepared = nodes.preProcessing_Interview("synthetic.pdf")
        self.assertEqual(prepared["resume_keywords"], ["Python", "협업"])
        self.assertEqual(prepared["current_question"], "첫 질문")
        self.assertEqual(prepared["question_queue"], ["둘째 질문"])
        self.assertEqual(prepared["remaining_topics"], ["협업"])
        self.assertEqual(prepared["next_step"], "evaluate")
        with patch.object(nodes, "extract_text_from_file", return_value=" "):
            with self.assertRaises(ValueError): nodes.preProcessing_Interview("empty.pdf")


if __name__ == "__main__":
    unittest.main()
