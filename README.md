# data_Cpr_Rstr
데이터 압축 및 복원 프로그램

## 128 Hz 재샘플링 스크립트

`resample_waveforms.py` 스크립트를 이용하면 오실로스코프 CSV 데이터를 1초에
128 샘플의 균등 간격으로 변환할 수 있습니다. 기본 설정은 D열(시간)과
E열(계측값)을 사용하도록 맞춰져 있으며, 필요할 경우 `--time-column`과
`--value-column` 옵션으로 다른 열을 지정할 수 있습니다.

```bash
python resample_waveforms.py 채널1.csv 채널1_128hz.csv --sample-rate 128
python resample_waveforms.py 채널2.csv 채널2_128hz.csv --sample-rate 128
python resample_waveforms.py 채널3.csv 채널3_128hz.csv --sample-rate 128
```

출력 파일에는 `time`과 `value` 두 열만 포함되며, 각각 균일한 시간축과 그에
해당하는 계측값을 나타냅니다.

## 3채널 주기 압축

`compress_waveforms.py` 스크립트는 128샘플짜리 주기를 기준으로 3개의 CSV 입력을
동시에 읽어 압축합니다. 파형의 형태가 변하지 않고 크기만 변하는 정상 구간은
대표파형과 주기별 배율만 저장되고, 세 채널 중 하나라도 `--event-threshold`를
넘는 순간 이상 구간으로 분류되어 앞·뒤 각각 3주기(`--boundary-cycles`)를 원본
그대로 보존합니다. 구간 내부는 변동이 작으면 정상 구간과 동일하게 배율로
저장되고, 큰 변동이 지속되면 전체 주기를 그대로 기록합니다.

```bash
python compress_waveforms.py ch1_128hz.csv ch2_128hz.csv ch3_128hz.csv \
    compressed.json --channels ch1 ch2 ch3 --value-columns value value value \
    --samples-per-cycle 128 --sample-rate 128
```

JSON 결과에는 메타데이터, 대표파형, 그리고 각 주기별로 어떤 방식으로 저장되었는지
정보가 담깁니다. `--normal-threshold`, `--event-threshold`, `--raw-threshold` 값을
조절하면 정상/이상 판단 기준을 쉽게 바꿀 수 있습니다.

## 압축 데이터 복원

`decompress_waveforms.py`는 위에서 생성한 JSON을 다시 128샘플 해상도의 CSV로
복원합니다. 대표파형과 배율 정보를 사용하여 정상 구간을 재구성하고, 압축 과정에서
원본을 그대로 보관했던 구간은 그대로 이어 붙입니다.

```bash
python decompress_waveforms.py compressed.json restored.csv
```

필요하다면 `--time-column` 옵션으로 시간 열 이름을 바꿀 수 있습니다.

## GUI 실행 및 EXE 패키징

`waveform_tool_gui.py`는 위의 세 스크립트를 하나의 그래픽 도구로 묶어 제공합니다.
세 개의 탭이 있으며 원본 CSV 업로드 → 재샘플링 → 압축 → 복원이 모두 가능합니다.

```bash
python waveform_tool_gui.py
```

독립 실행형 EXE가 필요하다면 PyInstaller를 사용하세요.

```bash
pip install -r requirements.txt
pyinstaller --onefile --windowed waveform_tool_gui.py
```
