from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import base64
import io
from model import build_model, EMOTIONS

app = Flask(__name__)

model = build_model()
model.load_state_dict(torch.load('emotion_model.pth', map_location='cpu'))
model.eval()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = base64.b64decode(data['image'].split(',')[1])
    pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
    frame = np.array(pil_img)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

    if len(faces) == 0:
        return jsonify({'faces': [], 'message': 'No face detected'})

    results = []
    for (x, y, w, h) in faces:
        face_crop = pil_img.crop((x, y, x+w, y+h))
        tensor = transform(face_crop).unsqueeze(0)

        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]

        scores = {EMOTIONS[i]: round(float(probs[i]), 3) for i in range(len(EMOTIONS))}
        dominant = max(scores, key=scores.get)
        results.append({'box': [int(x), int(y), int(w), int(h)], 'emotions': scores, 'dominant': dominant})

    return jsonify({'faces': results})

if __name__ == '__main__':
    app.run(debug=True)