# main.py

import os
import json
from agent_nodes import preProcessing_Interview, graph, InterviewState
from utils import summarize_interview_report

# --- API Key Load (Colab 환경에서는 필수) ---
def load_api_keys(filepath="api_key.txt"):
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

# NOTE: 실제 GitHub 사용 시 경로는 사용자가 설정해야 합니다.
# Colab 환경을 가정하고 임시 경로를 사용합니다.
path = './' # 현재 경로 가정
file_path = path + 'Resume_sample.pdf' # 실제 파일 경로로 변경 필요

try:
    # 1. 사전 준비 작업 실행
    print(f"--- '{file_path}' 파일로 사전 준비 작업 테스트 시작 ---")
    
    # NOTE: 실제 실행 시, API key 로드 함수가 필요합니다. 
    # Colab 노트북의 앞부분 코드를 참고하여 API 키 로드를 먼저 수행해야 합니다.
    # load_api_keys(path + 'api_key.txt') 

    state: InterviewState = preProcessing_Interview(file_path)
    
    print("--- [사전 준비 완료] ---")
    print(f"1. 첫 번째 질문: {state['current_question']}")
    print(f"2. 남은 주제: {state['remaining_topics']}")
    print("-" * 40)

    # 2. 사용자 응답 루프 실행
    while True:
        # 1) 질문 출력
        print("\n[질문]")
        print(state["current_question"])

        # 2) 답변 입력
        user_answer = input("\n[답변 입력]:\n")
        
        # 3) State에 답변 저장 (Agent 실행 전 수동 업데이트)
        state["current_answer"] = user_answer.strip()
        
        # 4) 그래프 실행: 평가 → 판단 → 다음 질문 생성 or 종료
        # NOTE: LangGraph의 첫 시작은 'evaluate' 노드가 됩니다.
        state = graph.invoke(state)
        
        # 5) 종료 조건 검사
        if state["next_step"] == "summarize":
            # 최종 보고서 요약은 summarize 노드에서 이미 처리됨
            break
            
# 3. 최종 보고서 출력 (노트북에서 시각화를 위해 함수를 별도로 호출)
except FileNotFoundError:
    print(f"[ERROR] 파일을 찾을 수 없습니다: {file_path}")
except Exception as e:
    print(f"[FATAL ERROR] Agent 실행 중 치명적인 오류 발생: {e}")
    
# 최종 보고서 재호출 (Agent가 END 노드에 도달했으므로)
if 'conversation' in state and state['conversation']:
    summarize_interview_report(state)
