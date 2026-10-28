"""PDF/DOCX 이력서로 실행하는 콘솔 면접."""
import argparse
import os
from pathlib import Path
from agent_nodes import preProcessing_Interview, graph
from utils import summarize_interview_report


def load_api_keys(filepath):
    """선택적 KEY=VALUE 파일. 기존 환경 변수를 우선하며 값은 출력하지 않습니다."""
    for line in Path(filepath).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def run_interview(state, *, graph_runner=graph, input_fn=input, output_fn=print):
    """사용자 입력과 그래프 실행 사이에 명시적인 턴 경계를 둡니다."""
    while state.get("next_step") != "summarize":
        output_fn("\n[질문]\n" + state["current_question"])
        answer = input_fn("\n[답변 입력 · /quit 종료]:\n").strip()
        if answer == "/quit":
            return {**state, "next_step": "summarize"}
        if not answer:
            output_fn("빈 답변입니다. 내용을 입력해 주세요.")
            continue
        state = graph_runner.invoke({**state, "current_answer": answer})
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description="이력서 기반 AI 면접 연습")
    parser.add_argument("resume", type=Path, help="PDF 또는 DOCX 이력서 경로")
    parser.add_argument("--key-file", type=Path, help="선택적 KEY=VALUE 형식 파일")
    args = parser.parse_args(argv)
    try:
        if args.key_file:
            load_api_keys(args.key_file)
        if not args.resume.is_file():
            raise FileNotFoundError("이력서 파일을 찾을 수 없습니다.")
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY 환경 변수를 설정해 주세요.")
        state = preProcessing_Interview(str(args.resume))
        state = run_interview(state)
        summarize_interview_report(state)
        return 0
    except (EOFError, KeyboardInterrupt):
        print("\n면접을 중단했습니다.")
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"[입력 확인] {exc}")
        return 1
    except Exception:
        print("[실행 오류] 모델 연결 또는 응답을 처리하지 못했습니다. 네트워크·인증·입력을 확인해 주세요.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
