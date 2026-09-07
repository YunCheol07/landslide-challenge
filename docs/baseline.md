# 첫 기준 모델: NDVI 차이 + 사전 경계

외부 데이터·사전학습 가중치 없이 새로 작성한 간단한 기준 모델이다. 제공 SAM3 predict.py의 내부/주변 NDVI 비교 아이디어는 참고했지만 코드를 복사하거나 실행하지 않았다. SAM3, best_decoder.pt, 외부 자산은 사용하지 않았다.

## 동작

1. 4밴드 NPY와 sites.json만 읽는다. 반사율은 값/10000이며 오프셋을 재보정하지 않는다.
2. 사전 폴리곤 내부와 경계 바깥 3~8px 링에 해당하는 화소 중심을 메모리에서 선택한다. 링에서는 모든 사전 개소 영역을 제외한다. 화소 중심이 정확히 경계 위이면 contains_xy에 의해 제외된다.
3. Red>0 & NIR>0인 화소에서 NDVI=(NIR-Red)/(NIR+Red)를 구한다. 특징은 내부 평균 NDVI - 링 평균 NDVI다. float64로 계산해 uint16 차감 오버플로를 피한다.
4. 특징이 임계값 이하면 존재로 판정한다. 임계값 후보는 학습 특징의 고유값과 최솟값 바로 아래 값이며, 학습 macro-F1 최대값을 선택한다. 동점은 작은 임계값을 선택한다. 검증 점수로 재조정하지 않는다.
5. 내부 또는 링의 유효 화소가 1개 미만이면 학습 라벨의 다수 클래스로 판정한다. 동수면 음성이다. 이 규칙과 링 폭은 첫 실험 전에 configs/baseline.json에 고정했다.
6. 존재이면 사전 폴리곤을 그대로 출력하고, 부재이면 빈 CSV 필드를 출력한다. 경계 학습·정교화는 없다.

학습 라벨은 고정 학습 scene의 명시적 visible:0/1만 사용하며 누락·ignore는 제외한다. 자기 교차 정답도 존재 라벨은 유지하므로 학습에 사용할 수 있다. 정답 폴리곤은 특징 추출이나 예측 경계에 쓰지 않는다. 추론 함수는 labels를 읽지 않는다. 분할은 라벨 개수 기반으로 사전에 고정되었지만 모델 임계값 학습은 학습 분할만 사용한다.

## 실행 및 로컬 산출물

```bash
.venv/bin/python scripts/run_baseline.py
# 새 코드/설정 실험은 새 이름을 부여하고 변경점을 하나만 기록
.venv/bin/python scripts/run_baseline.py --run-name baseline_v2
.venv/bin/python scripts/evaluate.py --predictions private/baseline_v1/predictions.csv --diagnostic
```

첫 명령은 이미 baseline_v1 결과가 있으면 덮어쓰지 않고 실패한다. private/<run-name>/model.json, predictions.csv, metrics.json에 모델 파라미터·검증 예측·집계를 저장한다. 모두 Git 제외다. 영상·마스크·패치·특징 캐시는 저장하지 않는다. 공개 실험 표에는 집계만 기록한다.

별도 추론 CLI:

```bash
.venv/bin/python scripts/predict_baseline.py \
  --input-dir /absolute/path/to/input \
  --model private/baseline_v1/model.json \
  --output private/inference.csv
```

입력은 창별 bands/와 sites.json이 있는 상위 폴더다. 레이블·분할 manifest·네트워크가 필요 없다. CSV 출력은 기존 파일을 덮어쓰지 않고 입력 폴더 안에는 쓰지 못하도록 한다. 입력 경로는 실제 로컬 경로로 바꿔야 한다. 서버 제출 노트북과 AIF 환경변수 어댑터는 아직 구현하지 않았다.

## 첫 검증 결과

| 항목 | 결과 |
|---|---:|
| 학습 라벨 수 | 1,516 |
| 검증 라벨 수 | 85 |
| 검증 예측 행 수(누락 슬롯 포함) | 91 |
| 존재 macro-F1 | 0.7438656909 |
| TP / TN / FP / FN | 40 / 24 / 3 / 18 |
| 유효 정답만의 진단용 개소 mIoU | 0.2323780260 |
| 공식 호환 형상·종합 점수 | 미확정 |
| 검증 특징 추출 시간(로컬, CSV/평가 제외) | 약 0.53초 |
| 유효 화소 부족 특징: 학습 / 검증 | 0 / 0 |

형상 진단값은 미확정 정답 2건을 제외한 값으로 공식 mIoU가 아니다. 전체 검증 라벨은 데이터의 5.31%이며 이 결과만으로 다른 scene·Private 성능을 추정하지 않는다. 서버 속도도 측정하지 않았다. 모델은 경계를 그대로 출력하므로 존재 판정에 성공해도 형상 정확도는 제한될 수 있다.

seed 42, 설정, 실행 시 HEAD commit, 작업 트리 변경 여부, 소스 SHA-256, 집계 점수와 시간을 기록했다. 실행 당시 작업 트리에 미커밋 코드가 있으므로 HEAD만으로 이 결과를 재현할 수 없으며 source_sha256을 함께 참조해야 한다. 코드를 확정 커밋한 뒤에는 해당 커밋에서 재실행해 재현성을 확인해야 한다.

## 검증

전체 21개 테스트 통과. 추가 테스트는 NDVI 부호·uint16 오버플로 방지, 무효 화소 처리, 누락·ignore 학습 제외, labels 없이 추론 및 손상 labels 무시, CSV 왕복과 기존 출력 보호를 확인한다. 실제 검증 CSV도 별도 평가 CLI로 읽어 집계 결과가 일치하는지 확인했다.
