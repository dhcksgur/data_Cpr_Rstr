# data_Cpr_Rstr
데이터 압축 및 복원 프로그램

## 60 Hz × 128 샘플 재샘플링 스크립트

`resample_waveforms.py` 스크립트를 이용하면 오실로스코프 CSV 데이터를 60 Hz
정현파 한 주기를 128 샘플로 표현할 수 있도록 (즉, 1초에 7,680 샘플) 균등
간격으로 변환할 수 있습니다. 기본 설정은 D열(시간)과 E열(계측값)을 사용하며
필요할 경우 `--time-column`과 `--value-column` 옵션으로 다른 열을 지정할 수
있습니다.

```bash
python resample_waveforms.py 채널1.csv 채널1_128samples.csv
python resample_waveforms.py 채널2.csv 채널2_128samples.csv
python resample_waveforms.py 채널3.csv 채널3_128samples.csv
```

만약 50 Hz 전원이나 다른 샘플 개수를 사용하고 싶다면 다음과 같이 지정할 수
있습니다.

```bash
python resample_waveforms.py 채널1.csv ch1_50hz.csv \
    --line-frequency 50 --samples-per-cycle 128
```

1초당 샘플 수를 직접 지정하고 싶은 경우 기존과 동일하게 `--sample-rate`를
사용하면 됩니다.

입력/출력 파일 이름에 공백이나 `-`가 포함된 경우에는 옵션 형태로 경로를
전달하면 인자 구분 문제가 생기지 않습니다.

```bash
python resample_waveforms.py --input "before after --00000.csv" \
    --output "before_after_128samples.csv"
```

출력 파일에는 `time`과 `value` 두 열만 포함되며, 각각 균일한 시간축과 그에
해당하는 계측값을 나타냅니다.
