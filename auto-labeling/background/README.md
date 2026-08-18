# Background 손 자동 어노테이션

Grounding DINO로 손 후보를 찾고 SAM3로 마스크와 bbox를 정제하는 손 전용
파이프라인이다. 한 프레임에는 손이 0개, 1개 또는 2개까지 들어갈 수 있다.
손이 없는 배경 프레임은 빈 라벨로 유지한다.

## 카메라 모드

기본 모드는 ZIP에 있는 물리 카메라를 모두 유지한다. 예를 들어 `cam_0`,
`cam_4`, `cam_6`는 각각 `cam0`, `cam4`, `cam6`로 저장된다.

`--camera-map`과 `--mapped-only`를 함께 지정하면 선택한 카메라만 이름을
바꿔 출력할 수 있다. `20260722_12.zip`의 검수 패키지는 다음 매핑을 쓴다.

- 원본 `cam_4` -> 최종 `cam0`: 상단 카메라
- 원본 `cam_6` -> 최종 `cam2`: 측면 카메라
- 원본 `cam_0`: 최종 패키지에서 제외

촬영 폴더별 프레임은 `<촬영순번>_<프레임순번>.jpg`로 합쳐진다. 예:
`1_1.jpg`, `2_1.jpg`.

## 실행

```bash
python auto-labeling/background/run_background.py \
  --minio-object chaiimage/etc/background/20260722_12.zip \
  --dataset-name 20260722_12_HAND_BACKGROUND \
  --camera-map cam_4=cam0 cam_6=cam2 \
  --mapped-only \
  --gpus 0 1 \
  --overwrite
```

## 최종 결과

```text
runs/sam3_with_dino/background/20260722_12_HAND_BACKGROUND/
├── cam0/
├── cam2/
└── annotations_hand_background.json
```

`annotations_hand_background.json`은 두 카메라를 합친 COCO Detection 및
Segmentation 형식이다. 카테고리는 `hand` 하나뿐이다.

중간 결과는 다음 경로에 저장한다.

```text
runs/sam3_with_dino/background/_work/20260722_12_HAND_BACKGROUND/
├── cam0/{labels,labels_seg,visualized,debug}
└── cam2/{labels,labels_seg,visualized,debug}
```
