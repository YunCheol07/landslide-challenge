# Landslide Challenge 2026

Sentinel-2 위성영상에서 **알려진 산사태 개소의 시점별 흔적 존재 여부와 경계**를 추정하는 대회 참가 프로젝트입니다.
2026 국립공원 위성 모니터링 AI 챌린지의 산사태 주제를 다룹니다.

> 현재 단계: 데이터 점검 → 고정 scene 분할 → 로컬 평가기 → 첫 기준 모델 → 오류 분석 완료. 서버 제출은 아직 수행하지 않았습니다.

## 핵심 결과 — 2026-09-08 정리

| 항목 | 확인 결과 |
|---|---|
| 데이터 점검 | 7개 scene, 15개 창, 영상 345개. 누락 라벨 90건 제외 |
| 고정 분할 | 학습 5개 scene / 검증 2개 scene, seed 42 |
| 첫 모델 | 내부·주변 NDVI 차이로 존재 판정, 사전 폴리곤으로 경계 출력 |
| 학습 방식 | 외부 가중치 없이 학습 분할에서만 임계값 선택 |
| 로컬 존재 macro-F1 | **0.7439** — 검증 양성 58건·음성 27건 |
| 검증 오류 | 미탐 18건, 오탐 3건 |
| 테스트 | 합성 테스트 23개 통과 |

**해석 범위:** 검증 라벨은 전체의 5.31%로 작고, 오류 분석 이후 개발용 검증으로 사용합니다. 자기 교차 정답 3건(검증 2건)의 공식 처리 기준이 미확정이므로 형상·종합 점수는 산출하지 않았습니다. 공식 서버 동등성·서버 성능·Private 성능은 확인하지 않았습니다.

**오류 분석:** 사전 경계와 정답의 겹침이 낮은 양성에서 미탐이 많았고, 오탐 3건은 한 개소에 집중됐습니다. 인과관계를 확정한 결과는 아닙니다. 다음 후보는 특징 추출 영역만 2px 확장하는 비교 실험이며 아직 실행하지 않았습니다.

[모델 및 결과](docs/baseline.md) · [오류 분석](docs/error-analysis.md) · [집계 실험 기록](experiments/results.csv)

## 문제와 설계 방향
- 입력: 256×256 RGB/NIR 위성 패치와 개소별 사전 폴리곤.
- 출력: 개소·날짜별 존재 판정과 픽셀 좌표 경계.
- 평가: 존재 macro-F1 60% + 형상 mIoU 40%.
- 검증: 동일 scene을 학습·검증에 섞지 않도록 분할하고, 누락/ignore 라벨을 제외합니다.
- 개선 순서: 제출 가능한 기준 모델 → 존재 판정 → 경계 정확도 → 실행 안정성.
- 제공 SAM3 베이스라인은 검토 대상입니다. 아직 사용하지 않았고 자체 개발 모델로 주장하지 않습니다.

## 진행 현황
- [x] Git 추적, 규칙·출처 기록, 실험·제출 기록 양식
- [x] 공개 저장소 검사 스크립트 및 로컬 push 훅
- [x] 데이터 점검 스크립트 및 실제 데이터 집계 검증
- [x] 고정 scene 검증 분할 및 변경 감지
- [x] 사용자 제공 규칙 기반 로컬 평가기 및 합성 테스트
- [ ] 공식 서버 동등성 및 자기 교차 정답 처리 기준 확인
- [ ] 대회 입력/출력 어댑터와 오프라인 제출 노트북
- [x] 첫 기준 모델: 학습 분할에서 임계값 선택 → 예측 CSV → 로컬 검증
- [x] 첫 모델 오류 분석: 공간 겹침·NDVI 대비·개소 집중도 확인
- [ ] 존재 판정 및 경계 개선
- [ ] 정상 서버 제출 및 재현 문서

## 저장소 구성
| 경로 | 목적 |
|---|---|
| src/ | NDVI 기준 모델·로컬 평가 구현 |
| configs/ | 경로 예시 및 실험 설정; 실제 경로는 Git 제외 |
| configs/split.json · scripts/make_split.py | 고정 분할 명세·생성·변경 감지 |
| experiments/results.csv | 공개 가능한 집계 실험 지표 |
| logs/submissions.csv | 제출 기록 양식; 응답 원문·키 제외 |
| docs/ | 규칙 요약·로드맵·출처·공개 정책 |
| submission/ | 제출물 구성 안내 |
| scripts/check_public_repo.py | Git 전체 이력의 공개 제외 항목 검사 |

## 로컬 준비
Python 3 표준 라이브러리만으로 저장소 검사를 실행할 수 있습니다.

```bash
cp configs/local.example.json configs/local.json
# local.json에 사용 권한이 있는 로컬 데이터 경로를 입력합니다.
python3 scripts/check_public_repo.py
```

제공 데이터는 기존 위치에서 읽기 전용으로 참조합니다. 이 저장소에 데이터를 복사하거나 업로드하지 않습니다.
데이터 점검 실행(Python 3.9 이상):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-audit.txt
.venv/bin/python scripts/audit_data.py
.venv/bin/python -m unittest discover -s tests -v
```

점검은 파일을 읽어 집계 JSON만 표준 출력에 내보냅니다. 원본·파생 영상이나 캐시를 저장하지 않습니다.
오류 발견 시 종료 코드 1, 설정 오류 시 2를 반환합니다. 현재 데이터는 정답 폴리곤 유효성 문제 3건으로 종료 코드 1이 예상됩니다.
검사 범위와 해석은 [데이터 점검 문서](docs/data-audit.md)를 참고하세요.

고정 검증 분할 생성·확인:

```bash
.venv/bin/python scripts/make_split.py
.venv/bin/python scripts/make_split.py --check
```

분할 배정은 Git 제외 경로에만 저장합니다. 데이터 편중으로 검증 라벨은 전체의 5.31%입니다.
[분할 방법과 도형 처리 기준](docs/validation.md)에 한계를 기록했습니다.

로컬 평가: `.venv/bin/python scripts/evaluate.py --predictions private/predictions.csv`

[평가 방법과 실행 검증](docs/evaluation.md)을 참고하세요. 미확정 정답 도형이 있으면 형상·종합 점수는 산출하지 않습니다.

첫 모델 학습·검증과 오류 분석:

```bash
.venv/bin/python scripts/run_baseline.py
.venv/bin/python scripts/analyze_baseline.py
```

기존 baseline_v1 결과는 덮어쓰지 않습니다. 모델·예측·로컬 분할은 Git 제외 private/에만 저장합니다. 현재 모델 의존성은 requirements-audit.txt에 고정되어 있습니다. 학습 실행 시 HEAD와 소스 해시를 기록하며, 첫 결과는 미커밋 작업 트리에서 측정했음을 실험 표에 명시했습니다.

## 공개 범위와 출처
코드·설계·작업 과정 중심의 포트폴리오입니다. 대회 영상, 라벨, 개소 좌표, 파생 이미지, 모델 가중치, 예측 원본, 인증정보를 배포하지 않습니다.
공개 검사는 자동 탐지 가능한 항목을 확인하며, 라이선스나 데이터 이용 조건에 대한 완전한 준수를 보증하지 않습니다.
외부 코드·가중치는 도입 전에 출처와 조건을 확인합니다. 저장소 전체에 임의의 오픈소스 라이선스를 적용하지 않았습니다.

- [대회 페이지](https://aifactory.space/ko/task/9305/data)
- [대회 규칙 요약](docs/competition.md) · [출처 기록](docs/provenance.md)
- [공개 정책](docs/publication.md) · [로드맵](docs/roadmap.md)

## 첫 기준 모델

`.venv/bin/python scripts/run_baseline.py`로 NDVI 기반 모델을 실행합니다. 기존 결과는 덮어쓰지 않습니다. [모델 구조·실행·결과·한계](docs/baseline.md)를 참고하세요. 외부 가중치 없이 학습 분할에서만 임계값을 선택하며, 경계는 사전 폴리곤을 사용합니다.

[첫 모델 오류 분석](docs/error-analysis.md): 미탐은 사전 경계와 정답의 낮은 겹침 및 약한 NDVI 대비와 연관됩니다. 오탐 3건은 한 개소에 집중되었습니다. 모델은 변경하지 않았으며 다음 후보는 공간 특징 범위만 확장하는 실험입니다.
