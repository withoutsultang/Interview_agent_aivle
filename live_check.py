"""Run two real transition calls on fictional answers for manual review."""
import os

from agent_nodes import next_topic_question
from demo import transition_state


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY repository secret is required.")
        return 1
    for answer in ("오류 로그를 비교해 설정 누락을 찾았습니다.",
                   "오류 원인을 아직 파악하지 못했습니다."):
        state = transition_state(answer)
        state["conversation"] = [{"question": state["current_question"], "answer": answer}]
        try:
            result = next_topic_question(state)
            question = result["current_question"]
            if result["current_strategy"] != "협업" or not question.strip() or "?" not in question:
                raise ValueError("invalid transition")
        except Exception as exc:
            print(f"Live check failed ({type(exc).__name__}).")
            return 1
        print(f"가상 답변: {answer}\n생성 질문: {question}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
