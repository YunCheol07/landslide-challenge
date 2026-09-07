# 코드 설명서 준비 메모

공식 Google Docs 서식은 아직 확인하지 못했다. 아래는 내부 기록용이며 제출 시 공식 서식으로 옮긴다.

- 과제와 데이터 처리:
- 고정 검증 분할 및 누수 방지: configs/split.json 및 docs/validation.md. scene 단위 학습 5 / 검증 2, seed 42. 로컬 manifest와 입력 해시로 변경 탐지.
- 모델 구조와 사전학습 자산 출처 (provenance.md 참조):
- 학습 환경·seed·패키지 버전·명령:
- 존재 판정 및 경계 후처리:
- 검증 성능과 최종 선택 근거: 첫 NDVI 기준 모델의 로컬 존재 macro-F1 0.7438656909(검증 85건). 형상·종합 점수 미확정. docs/baseline.md 참조. src/evaluation.py의 로컬 평가 구현 및 검증은 docs/evaluation.md 참조. 공식 서버 동등성 미검증.
- 추론 실행 방법·자원·시간:
- 재현 가능한 commit·가중치 체크섬:
- 외부 데이터/가중치의 라이선스 및 필요한 고지:
