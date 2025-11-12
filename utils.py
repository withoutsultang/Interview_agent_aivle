# utils.py

import os
import fitz  # PyMuPDF
from docx import Document
import json

# --- File Extraction & Parsing ---

def extract_text_from_file(file_path: str) -> str:
  """PDF 또는 DOCX 파일에서 텍스트를 추출합니다."""
  ext = os.path.splitext(file_path)[1].lower()
  if ext == ".pdf":
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text
  elif ext == ".docx":
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
  else:
    raise ValueError("지원하지 않는 파일 형식입니다. PDF 또는 DOCX만 허용됩니다.")

# --- Display/Report Helpers ---

def summarize_interview_report(state: dict):
    """최종 면접 피드백 보고서를 콘솔에 출력합니다."""
    print("\n--- [AI 면접 최종 피드백 보고서] ---")
    print("\n[이력서 요약]")
    print(state.get("resume_summary", "요약 정보 없음"))

    print("\n[이력서 핵심 키워드]")
    print(", ".join(state.get("resume_keywords", [])))

    print("\n" + "="*40)
    print(" [면접 상세 내용 및 평가]")
    print("="*40)

    conversation = state.get("conversation", [])
    evaluation = state.get("evaluation", [])

    if not conversation or not evaluation:
        print("\n면접 기록이 없습니다.")
        return

    for i, (conv, eval_item) in enumerate(zip(conversation, evaluation)):
        question = conv.get("question", "질문 없음")
        answer = conv.get("answer", "답변 없음")
        eval_dict = eval_item.get("evaluation", {})

        relevance = eval_dict.get("연관성", "평가 없음")
        specificity = eval_dict.get("구체성", "평가 없음")
        feedback = eval_dict.get("평가_의견", "평가 의견 없음")

        print(f"\n--- [질문 {i+1}] ---")
        print(f"Q: {question}")

        print("\n[지원자 답변]")
        print(f"A: {answer}")

        print("\n[AI 평가]")
        print(f"  - 질문 연관성: {relevance}")
        print(f"  - 답변 구체성: {specificity}")
        print(f"  - 평가 의견: {feedback}")
        print("-" * 30)

    print("\n--- [면접 종료] ---")

# NOTE: 필요하다면 PDF 차트 생성 관련 코드도 여기에 포함됩니다.
