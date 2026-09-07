# Landslide Challenge 2026

2026 국립공원 위성 모니터링 AI 챌린지 — 산사태 붕괴지 탐지 및 위험도 분석.

## 현재 상태
- 전용 프로젝트 및 로컬 Git 초기화. 원본 데이터·배포물은 복사하지 않음.
- 기존 읽기 전용 점검: 15개 창, 7개 scene ID, 345장, 창-개소 62개, 보임 642 / 안 보임 959, 누락 90건.
- 기준 모델 학습, SAM3 다운로드, 서버 제출은 아직 수행하지 않음.
- 받은 배포물은 predict.py + GeoTIFF/GeoJSON 샘플 방식. 대회용 predict.ipynb는 없음.
- 다음: 검증 분할·공식 평가 구현 → 제출 형식 어댑터 → 정상 채점 → 존재 판정 개선.
- 학습 GPU 환경, SAM3 접근 및 라이선스 검토, 공식 제출 노트북/SDK 설치 정보는 확인 필요.

## 구성
- configs/local.example.json: 로컬 입력 경로 예시. 필요하면 local.json으로 설정 복사(데이터 복사 아님).
- src/: 전처리·학습·평가·추론 코드
- splits/: scene별 분할 명세
- experiments/results.csv: 집계 실험 기록
- logs/submissions.csv: 팀 제출 기록
- docs/: 대회 규칙·로드맵·출처와 설명서 메모
- submission/: 최종 predict.ipynb와 assets/를 구성할 위치

## 관리
코드·문서·설정은 Git으로 관리하며 데이터·모델·키는 제외한다.
GitHub 비공개 저장소: https://github.com/YunCheol07/landslide-challenge
main은 origin/main을 추적한다. 데이터·가중치와 실행 출력은 push하지 않는다.
대회 규칙은 docs/competition.md, 작업 규칙은 AGENTS.md를 따른다.
