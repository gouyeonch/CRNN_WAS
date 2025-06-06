import torch
import torch.nn as nn
from pathlib import Path
from CRNN import CRNN
from MelSpectrogram import preprocess_mel

def predict(file_path, model_path="siren_crnn.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mel_tensor = preprocess_mel(file_path).to(device)

    print("🔎 입력 텐서 크기:", mel_tensor.shape)

    with torch.no_grad():
        output = model(mel_tensor)
        prob = torch.softmax(output, dim=1).cpu().numpy()[0]  # [normal, announcement, siren]
        print("🔢 softmax 확률 분포:", prob)
        if prob[2] > 0.6:
            print(f"[{Path(file_path).name}] → 🚨 사이렌 감지됨 (siren)")
        elif prob[1] > 0.6:
            print(f"[{Path(file_path).name}] → 📢 안내 방송 (announcement)")
        else:
            print(f"[{Path(file_path).name}] → ✅ 일반 소리 (normal)")
        # print("FC weight example:", model.fc.weight.data[:1])


if __name__ == '__main__':
    # 테스트 시 predict만 실행
    test_file = "test_siren.wav"
    predict(test_file)
