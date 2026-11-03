"""A repeatable interview using synthetic inputs and scripted model replies."""
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

import agent_nodes
from utils import summarize_interview_report


def transition_state(answer):
    return dict(
        current_question="배포 오류를 어떻게 해결했나요?",
        current_answer=answer,
        current_strategy="문제 해결",
        question_queue=[],
        remaining_topics=["협업"],
        question_strategy={"면접 질문 전략": {"협업": {
            "예시 질문": ["팀원에게 변경 사항을 어떻게 공유했나요?"]
        }}},
        generate_count=0,
        conversation=[],
        evaluation=[],
        next_step="evaluate",
    )


def run_demo(run_interview):
    print("[데모] 가상 사례 · 고정 모델 응답 · API 호출 없음")
    answers = iter(["오류 로그를 비교해 설정 누락을 찾았습니다.",
                    "수정 내용과 재현 방법을 팀 문서에 남겼습니다."])

    def scripted_input(prompt):
        answer = next(answers)
        print(prompt + answer)
        return answer

    grade = '{"연관성":"상","구체성":"중","평가_의견":"판단 근거를 한 가지 더 설명해 보세요."}'
    model = FakeListChatModel(responses=[
        grade, "next_topic",
        "설정 누락을 찾으셨군요. 팀원에게 변경 사항을 어떻게 공유했나요?",
        grade, "summarize",
    ])
    with patch.object(agent_nodes, "get_llm", return_value=model):
        result = run_interview(transition_state(""), input_fn=scripted_input)
    summarize_interview_report(result)
