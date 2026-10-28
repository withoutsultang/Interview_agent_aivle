# agent_nodes.py

import os
import json
from typing import Literal, TypedDict, List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser, CommaSeparatedListOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

# utils.py에서 추출 함수 임포트
from utils import extract_text_from_file

# --- 환경 설정 ---
LLM_MODEL = "gpt-4o-mini"
# OPENAI_API_KEY는 환경 변수에서 로드된다고 가정
def get_llm():
    """모듈 import 시에는 인증이나 네트워크 호출을 하지 않습니다."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY 환경 변수를 설정해 주세요.")
    return ChatOpenAI(model=LLM_MODEL, temperature=0)


MAX_TURNS = 5
MAX_FOLLOWUPS = 2

# --- State 정의 (모든 정보 포함) ---
class InterviewState(TypedDict):
    # 고정 정보
    resume_text: str
    resume_summary: str
    resume_keywords: List[str]
    question_strategy: Dict[str, Dict]
    
    # 면접 진행 상태
    current_question: str
    current_answer: str
    current_strategy: str # 현재 질문의 전략 카테고리 (예: '경력 및 경험')
    
    # --- 고도화 항목 ---
    question_queue: List[str] # 현재 주제의 남은 예시 질문 (Pop으로 소진)
    remaining_topics: List[str] # 남아있는 주제 카테고리 (Pop으로 소진)
    generate_count: int # 심화 질문 생성 횟수 카운터

    # 인터뷰 로그
    conversation: List[Dict[str, str]]
    evaluation : List[Dict[str, str]]
    next_step : str # "generate", "summarize", "next_topic", "end" 중 하나

# ===============================
# 🔹 Node Functions
# ===============================

def analyze_resume(state: InterviewState) -> InterviewState:
    """이력서 분석: 요약 및 키워드 추출"""
    resume_text = state.get('resume_text')
    if not resume_text:
      return {**state, "resume_summary": "이력서 텍스트 없음", "resume_keywords": []}

    # 1. 요약 추출 체인
    summary_prompt = ChatPromptTemplate.from_template(
        "\"\"\"당신은 전문 채용 담당자입니다. 다음 이력서 텍스트를 3-4줄의 핵심 내용으로 요약해 주세요.\n        ---\n        {resume}\n        \"\"\""
    )
    summary_chain = summary_prompt | get_llm() | StrOutputParser()
    resume_summary = summary_chain.invoke({"resume": resume_text})

    # 2. 키워드 추출 체인
    keywords_prompt = ChatPromptTemplate.from_template(
        "\"\"\"당신은 전문 IT 헤드헌터입니다. 다음 이력서 텍스트에서 가장 중요한 핵심 키워드 10개를 쉼표(,)로 구분하여 추출해 주세요.\n        예: Python, 데이터 분석, NLP, 프로젝트 관리, 리더십\n        ---\n        {resume}\n        \"\"\""
    )
    keywords_chain = keywords_prompt | get_llm() | CommaSeparatedListOutputParser()
    resume_keywords = keywords_chain.invoke({"resume": resume_text})

    return {
        **state,
        "resume_summary": resume_summary,
        "resume_keywords": resume_keywords,
    }

def normalize_strategy(value):
    if not isinstance(value, dict):
        raise ValueError("질문 전략은 JSON 객체여야 합니다.")
    topics = value.get("면접 질문 전략", value)
    if not isinstance(topics, dict):
        raise ValueError("면접 질문 전략의 주제 구조가 올바르지 않습니다.")
    clean = {}
    for topic, detail in topics.items():
        if not isinstance(topic, str) or not isinstance(detail, dict):
            continue
        questions = detail.get("예시 질문", [])
        if not isinstance(questions, list):
            continue
        questions = [q.strip() for q in questions if isinstance(q, str) and q.strip()]
        if questions:
            clean[topic] = {**detail, "예시 질문": questions}
    if not clean:
        raise ValueError("질문 전략에 유효한 예시 질문이 없습니다.")
    return {"면접 질문 전략": clean}


def generate_question_strategy(state: InterviewState) -> InterviewState:
    """질문 전략 수립: 3가지 분야의 질문 방향과 예시 질문을 JSON으로 생성"""
    resume_summary = state.get('resume_summary')
    resume_keywords = state.get('resume_keywords')
    
    parser = JsonOutputParser()

    prompt = ChatPromptTemplate.from_template(
        "\"\"\"당신은 AI 면접관의 질문 전략을 수립하는 전문가입니다.\n"
        "아래 이력서 요약과 핵심 키워드를 바탕으로 3가지 주요 카테고리(경력 및 경험, 커뮤니케이션 능력, 논리적 사고)에 대한 면접 질문 전략을 수립해 주세요.\n"
        "\n"
        "각 카테고리별로 \"질문 방향\"과 2개의 \"예시 질문\" 리스트를 포함해야 합니다.\n"
        "최상위 키는 반드시 면접 질문 전략이며, 그 값은 카테고리 이름을 키로 갖는 객체입니다.\n"
        "{format_instructions}\n"
        "\n"
        "--- 이력서 요약 ---\n"
        "{summary}\n"
        "\n"
        "--- 핵심 키워드 ---\n"
        "{keywords}\n"
        "\"\"\"",
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )

    chain = prompt | get_llm() | parser
    
    strategy_dict = chain.invoke({'summary': resume_summary, 'keywords': resume_keywords})
    strategy_dict = normalize_strategy(strategy_dict)

    
    # 남아있는 주제 카테고리 초기화
    remaining_topics = list(strategy_dict.get("면접 질문 전략", {}).keys())
    
    # 첫 질문 및 큐 초기화 (경력 및 경험 카테고리로 시작)
    first_topic = "경력 및 경험" if "경력 및 경험" in remaining_topics else remaining_topics[0]
    first_question = ""
    question_queue = []
    current_strategy = ""
    
    if first_topic in strategy_dict.get("면접 질문 전략", {}):
        questions = strategy_dict["면접 질문 전략"][first_topic].get("예시 질문", [])
        if questions:
            first_question = questions[0]
            question_queue = questions[1:] # 첫 질문 제외한 나머지는 큐에
            current_strategy = first_topic
            # remaining_topics에서 첫 주제 제거
            if first_topic in remaining_topics:
                remaining_topics.remove(first_topic)


    return {
        **state,
        "question_strategy": strategy_dict,
        "current_question": first_question if first_question else "자기소개 부탁드립니다.",
        "current_strategy": current_strategy if current_strategy else "자유 주제",
        "question_queue": question_queue,
        "remaining_topics": remaining_topics,
        "generate_count": 0,
    }


def evaluate_answer(state: InterviewState) -> InterviewState:
    """답변 평가: 질문과의 연관성, 구체성 등 2개 항목으로 LLM 평가 수행 (점수 및 의견 포함)"""
    current_question = state.get("current_question")
    current_answer = state.get("current_answer")
    
    # conversation 업데이트는 여기서 수행 (질문과 답변이 짝지어질 때)
    if not current_question or not current_question.strip():
        raise ValueError("평가할 질문이 없습니다.")
    if not current_answer or not current_answer.strip():
        raise ValueError("답변을 입력해 주세요.")
    conversation = list(state.get("conversation", []))
    conversation.append({"question": current_question, "answer": current_answer})
    
    evaluation = list(state.get("evaluation", []))
    
    # NOTE: 미션2에서 평가 항목을 5개로 고도화할 때 이 프롬프트 수정 필요
    prompt = ChatPromptTemplate.from_template(
        "\"\"\"당신은 AI 면접관의 답변 평가 전문가입니다.\n"
        "주어진 면접 질문과 지원자의 답변을 바탕으로 다음 두 가지 항목에 대해 '상', '중', '하' 중 하나로 평가하고 간단한 평가 의견을 작성하여 JSON 형식으로 반환해 주세요.\n"
        "\n"
        "JSON 형식은 다음과 같아야 합니다:\n"
        "{{\n"
        "    \"연관성\": \"'상', '중', '하' 중 하나\",\n"
        "    \"구체성\": \"'상', '중', '하' 중 하나\",\n"
        "    \"평가_의견\": \"간단한 평가 의견 (문자열)\"\n"
        "}}\n"
        "\n"
        "평가 기준:\n"
        "1. 질문과의 연관성: 질문의 의도를 정확히 파악하고 관련 내용을 답변했는가?\n"
        "2. 답변의 구체성: 경험이나 생각을 구체적인 사례나 근거를 들어 설명했는가?\n"
        "\n"
        "--- 면접 질문 ---\n"
        "{question}\n"
        "\n"
        "--- 지원자 답변 ---\n"
        "{answer}\n"
        "\"\"\""
    )
    
    chain = prompt | get_llm() | JsonOutputParser()

    eval_result = chain.invoke({"question": current_question, "answer": current_answer})
    if not isinstance(eval_result, dict) or any(
        eval_result.get(key) not in {"상", "중", "하"} for key in ("연관성", "구체성")
    ):
        raise ValueError("답변 평가 형식이 올바르지 않습니다.")

    
    # 평가 기록 업데이트
    evaluation.append({"question": current_question, "answer": current_answer, "evaluation": eval_result})
    
    return {
        **state,
        "conversation": conversation, # 대화 기록 업데이트
        "evaluation": evaluation
    }


def decide_next_step(state: InterviewState) -> InterviewState:
    """다음 단계 결정 (종료, 주제 전환, 다음 질문, 심화 질문)"""
    
    conversation_count = len(state.get("conversation", []))
    question_queue = state.get("question_queue", [])
    remaining_topics = state.get("remaining_topics", [])
    evaluation = state.get("evaluation", [])
    generate_count = state.get("generate_count", 0)

    if conversation_count >= MAX_TURNS:
        return {**state, "next_step": "summarize"}

    # 1. LLM 호출 준비: 최근 평가 결과와 질문 기록 정리
    if not evaluation:
        return {**state, "next_step": "generate"} # 평가가 없으면 일단 질문 생성으로 보냄
        
    last_eval_item = evaluation[-1]
    last_evaluation_dict = last_eval_item.get("evaluation", {})
    
    # 연관성/구체성 등급을 문자열로 추출 (LLM에게 전달하기 위함)
    eval_summary = (
        f"연관성: {last_evaluation_dict.get('연관성', '중')}, "
        f"구체성: {last_evaluation_dict.get('구체성', '중')}"
    )

    # LLM 프롬프트
    prompt = ChatPromptTemplate.from_template(
        "\"\"\"당신은 AI 면접 진행 관리자입니다.\n"
        "지원자의 최근 답변 평가, 남은 예시 질문 개수, 남은 주제 목록을 보고 다음 행동을 결정해 주세요.\n"
        "\n"
        "[결정 규칙]\n"
        "1. (최우선) 현재 대화 횟수가 {count}인데 최대 5번을 넘으면 안됩니다. 5번 이상이면 'summarize'를 반환하세요.\n"
        "2. (2순위: 심화 질문) **최근 답변 평가의 '연관성' 또는 '구체성'이 '하' 이면:** 이 주제를 심화하기 위해 'generate'를 반환합니다. (심화 질문 생성 노드)\n"
        "3. (3순위: 다음 예시 질문) **현재 주제의 '남은 예시 질문'이 [있음]** 이면: 'next_question'을 반환합니다.\n"
        "4. (4순위: 주제 전환) **2, 3번 규칙에 해당하지 않고, 남은 주제 목록이 [있음]** 이라면: 다음 주제로 넘어가기 위해 'next_topic'을 반환합니다. (주제 전환 노드)\n"
        "5. (5순위: 종료) **남은 주제 목록이 [없음]** 이라면: 'summarize'를 반환합니다.\n"
        "\n"
        "당신의 결정은 오직 'generate', 'next_question', 'next_topic', 'summarize' 네 단어 중 하나여야 합니다.\n"
        "\n"
        "--- 현재 상태 ---\n"
        "1. 현재 대화 횟수: {count}\n"
        "2. 최근 답변 평가: {eval_summary}\n"
        "3. 현재 주제의 남은 예시 질문 개수: {queue_count}\n"
        "4. 남은 주제 목록: {remaining_topics}\n"
        "\n"
        "--- 다음 행동 (오직 단어 하나로 응답) ---"
        "\"\"\""
    )
    
    chain = prompt | get_llm() | StrOutputParser()
    
    # LLM 호출
    next_action = chain.invoke({
        "count": conversation_count,
        "eval_summary": eval_summary,
        "queue_count": len(question_queue),
        "remaining_topics": remaining_topics
    }).strip().lower()

    fallback = "next_question" if question_queue else "next_topic" if remaining_topics else "summarize"
    if next_action not in {"generate", "next_question", "next_topic", "summarize"}:
        next_action = fallback
    if next_action == "generate" and generate_count >= MAX_FOLLOWUPS:
        next_action = fallback
    if next_action == "next_question" and not question_queue:
        next_action = "next_topic" if remaining_topics else "summarize"
    if next_action == "next_topic" and not remaining_topics:
        next_action = "next_question" if question_queue else "summarize"
    return {
        **state,
        "generate_count": generate_count + 1 if next_action == "generate" else generate_count,
        "next_step": next_action,
    }


def next_topic_question(state: InterviewState) -> InterviewState:
    """새 주제의 첫 질문을 선택하고 current_question에 설정"""
    
    question_queue = list(state.get("question_queue", []))
    remaining_topics = list(state.get("remaining_topics", []))
    strategy = state.get("question_strategy", {})

    changing_topic = False
    # 1. 현재 주제의 남은 질문이 있다면 그것을 먼저 소진 (Safety Check)
    if question_queue:
        new_question = question_queue.pop(0)
        current_strategy = state.get("current_strategy")

    # 2. 다음 주제로 전환해야 할 경우
    elif remaining_topics:
        changing_topic = True
        current_topic_name = remaining_topics.pop(0) # 다음 주제를 꺼냄
        current_strategy = current_topic_name
        
        # 새 주제의 질문 리스트를 가져와서 첫 질문을 선택하고 나머지는 큐에 넣음
        questions = strategy.get("면접 질문 전략", {}).get(current_topic_name, {}).get("예시 질문", [])
        if not questions:
            new_question = f"[{current_topic_name}] 주제에 대한 첫 질문입니다: 해당 주제 관련 경험을 설명해 주세요."
        else:
            new_question = questions[0]
            question_queue = questions[1:]
    
    else: # 남은 주제가 없음 (route_next에서 처리되어야 하지만 안전장치)
        return {
            **state, 
            "next_step": "summarize",
            "current_question": "면접을 종료합니다." # 종료 메시지
        }
    
    if changing_topic and state.get("conversation"):
        previous = state["conversation"][-1]
        transition = ChatPromptTemplate.from_template(
            "당신은 면접관입니다. 이전 답변에서 실제로 언급된 내용을 짧게 짚고 "
            "새 주제의 질문으로 자연스럽게 이어 주세요. 답변에 없는 경험이나 성과를 만들지 마세요. "
            "목표 질문의 의도를 유지하고, 중복 질문 없이 짧은 연결 문장과 질문 하나만 출력하세요.\n"
            "이전 질문: {question}\n이전 답변: {answer}\n새 주제: {topic}\n목표 질문: {target}"
        )
        connected = (transition | get_llm() | StrOutputParser()).invoke({
            "question": previous["question"], "answer": previous["answer"],
            "topic": current_strategy, "target": new_question,
        }).strip()
        if connected:
            new_question = connected

    # 3. State 업데이트
    return {
        **state,
        "current_question": new_question,
        "current_answer": "", # 답변은 초기화
        "current_strategy": current_strategy,
        "question_queue": question_queue,
        "remaining_topics": remaining_topics,
        "generate_count": 0, # 새 예시 질문마다 심화 횟수 초기화
        "next_step": "evaluate" # 다음 실행은 답변을 기다린 후 평가로 돌아가야 함
    }


def generate_question(state: InterviewState) -> InterviewState:
    """심화 질문 생성: 이전 평가/대화를 기반으로 더욱 심도 있는 질문을 LLM이 생성"""
    
    resume_summary = state.get("resume_summary", "")
    resume_keywords = state.get("resume_keywords", [])
    question_strategy = state.get("question_strategy", {})
    conversation = state.get("conversation", [])
    evaluation = state.get("evaluation", [])
    generate_count = state.get("generate_count", 0) # 심화 질문 횟수

    # 1. LLM이 읽기 쉬운 대화 기록 문자열 생성
    history_str = ""
    for i, (conv, eval_item) in enumerate(zip(conversation, evaluation)):
        history_str += f"\n--- 질문 {i+1} ---\n"
        history_str += f"Q: {conv['question']}\n"
        history_str += f"A: {conv['answer']}\n"
        history_str += f"평가: {json.dumps(eval_item['evaluation'], ensure_ascii=False)}\n"

    # 2. 심화 질문의 깊이 코멘트 (generate_count를 활용한 고도화)
    if generate_count <= 1:
        depth_comment = "최근 답변의 부족한 부분을 채우거나, 기술적 이해도를 더 확인할 수 있는 심화 질문을 생성하세요."
    elif generate_count == 2:
        depth_comment = "현재 답변의 논리나 경험의 구체성을 검증할 수 있는 더 깊은 질문을 생성하세요. (압박 질문 형태도 고려)"
    else: # 3회차 이상
        depth_comment = "지금까지의 답변을 종합하여, 지원자의 사고력, 문제 해결력, 가치관을 탐구할 수 있는 고난도 질문을 생성하세요."


    # 3. LLM 프롬프트
    prompt = ChatPromptTemplate.from_template(
        "\"\"\"당신은 전문 AI 면접관입니다.\n"
        "지원자의 역량을 심층적으로 파악하기 위한 **다음 심화 질문**을 하나 생성해 주세요.\n"
        "\n"
        "[규칙]\n"
        "1. **절대로** 이력서의 예시 질문이나 이전 면접 기록에 나온 질문을 그대로 반복하지 마세요.\n"
        "2. {depth_comment} \n"
        "3. 질문은 간결하고 명확하게 한 문장으로 만들어 주세요.\n"
        "\n"
        "--- [지원자 이력서 요약] ---\n"
        "{summary}\n"
        "\n"
        "--- [지금까지의 면접 기록 (질문, 답변, 평가)] ---\n"
        "{history}\n"
        "\n"
        "--- [다음 심화 질문 (오직 질문 하나만 생성)]:\n"
        "\"\"\""
    )
    
    chain = prompt | get_llm()
    
    # LLM 호출하여 다음 질문 생성
    response = chain.invoke({
        "summary": resume_summary,
        "history": history_str,
        "depth_comment": depth_comment
    })

    # 4. State 업데이트
    return {
        **state,
        "current_question": response.content.strip(),
        "current_answer": "", # 답변은 초기화
        "next_step": "evaluate" # 다음 실행은 답변을 기다린 후 평가로 돌아가야 함
    }


def preProcessing_Interview(file_path: str) -> InterviewState:
    """미션 1. 사전 준비 절차를 한 번에 실행하고 첫 질문을 설정합니다."""

    # 1. 텍스트 추출
    resume_text = extract_text_from_file(file_path)
    if not resume_text.strip():
        raise ValueError("이력서에 추출 가능한 텍스트가 없습니다.")

    # 2. State 초기화
    initial_state: InterviewState = {
        "resume_text": resume_text,
        "resume_summary": '',
        "resume_keywords": [],
        "question_strategy": {},
        "current_question": '',
        "current_answer": '',
        "current_strategy": '',
        "conversation": [],
        "evaluation": [],
        "next_step" : '',
        "question_queue": [],
        "remaining_topics": [],
        "generate_count": 0,
    }

    # 3. 이력서 분석
    state = analyze_resume(initial_state)

    # 4. 질문 전략 수립 및 첫 질문 설정 (generate_question_strategy에 통합)
    state = generate_question_strategy(state)
    
    # next_step은 초기 질문에 대한 답변을 받아야 하므로, 다음 노드는 'evaluate'가 되어야 함.
    state['next_step'] = 'evaluate' 

    return state

def summarize_interview(state: InterviewState) -> InterviewState:
    """종료 상태 설정. 보고서 출력은 호출자에서 한 번만 합니다."""
    return {**state, "next_step": "summarize", "current_answer": ""}


def route_next(state: InterviewState) -> Literal["next_question", "generate", "next_topic", "summarize"]:
    action = state.get("next_step", "summarize")
    return action if action in {"next_question", "generate", "next_topic", "summarize"} else "summarize"


def build_graph():
    """한 번의 invoke는 입력된 답변 한 개만 평가하고 다음 질문 또는 종료를 반환합니다."""
    workflow = StateGraph(InterviewState)
    workflow.add_node("evaluate", evaluate_answer)
    workflow.add_node("decide_next", decide_next_step)
    workflow.add_node("next_question", next_topic_question)
    workflow.add_node("next_topic", next_topic_question)
    workflow.add_node("generate", generate_question)
    workflow.add_node("summarize", summarize_interview)
    workflow.set_entry_point("evaluate")
    workflow.add_edge("evaluate", "decide_next")
    workflow.add_conditional_edges("decide_next", route_next, {
        name: name for name in ("next_question", "next_topic", "generate", "summarize")
    })
    for name in ("next_question", "next_topic", "generate", "summarize"):
        workflow.add_edge(name, END)
    return workflow.compile()


graph = build_graph()
