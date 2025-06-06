import sounddevice as sd
import soundfile as sf
import time

from DetectAudio import predict

DURATION = 5  # 녹음 길이 (초)
SAMPLE_RATE = 16000
LOOP_SLEEP = 1 # 실시간 루프 지연시간 (초)

TRUE_SAMPLE_PATH_1 = "./dataset/siren/siren_1_clip_0_orig.wav"
TRUE_SAMPLE_PATH_2 = "./dataset/siren/siren_1_clip_1_orig.wav"
TRUE_SAMPLE_PATH_3 = "./dataset/siren/siren_1_clip_2_orig.wav"

TRUE_SAMPLE_PATH_4 = "./dataset/announcement/announcement_clip_1_orig.wav"
TRUE_SAMPLE_PATH_5 = "./dataset/announcement/announcement_clip_2_orig.wav"
TRUE_SAMPLE_PATH_6 = "./dataset/announcement/announcement_clip_3_orig.wav"


def record_audio(filename, duration, samplerate):
    print("녹음 시작...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    sf.write(filename, audio, samplerate)
    print("녹음 저장 완료:", filename)

def live_siren_detection_loop():
    while True:
        print("\n새 입력 감지 중...")
        record_audio("test_input.wav", DURATION, SAMPLE_RATE)
        detected = predict("test_input.wav")
        # detected = predict(TRUE_SAMPLE_PATH_1)
        # detected = predict(TRUE_SAMPLE_PATH_2)
        # detected = predict(TRUE_SAMPLE_PATH_3)
        # detected = predict(TRUE_SAMPLE_PATH_4)
        # detected = predict(TRUE_SAMPLE_PATH_5)
        # detected = predict(TRUE_SAMPLE_PATH_6)

        if detected:
            print("실시간 사이렌 감지 완료!")
        else:
            print("감지 없음")
        time.sleep(LOOP_SLEEP)  # 1초 쉬고 다음 감지 시작

if __name__ == '__main__':
    live_siren_detection_loop()
