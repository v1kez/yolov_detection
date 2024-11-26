from flask import Flask, request, render_template, send_file
from PIL import Image
import torch
import cv2
import numpy as np
import os

app = Flask(__name__)

# Папка для сохранения загруженных изображений
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Загрузка модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


@app.route('/')
def home():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'Нет файла в запросе', 400

    file = request.files['file']

    if file.filename == '':
        return 'Нет выбранного файла', 400

    # Сохраняем файл
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Получаем выбранные классы
    classes = request.form.getlist('classes')
    filter_objects = request.form.get('filter_objects')  # Получаем значение фильтрации

    # Загрузка изображения
    image = Image.open(file_path)

    # Выполнение детекции объектов
    results = model(image)

    # Фильтрация результатов, если выбраны классы
    confidence_threshold = 0.5
    filtered_results = results.pandas().xyxy[0]

    if filter_objects:  # Если фильтрация включена
        filtered_results = filtered_results[
            (filtered_results['name'].isin(classes)) & (filtered_results['confidence'] >= confidence_threshold)]
    else:  # Если фильтрация отключена, используем все результаты
        filtered_results = filtered_results[filtered_results['confidence'] >= confidence_threshold]

    # Преобразование изображения в массив NumPy
    filtered_image = np.array(image)

    # Рисуем рамки и текст на изображении
    for _, row in filtered_results.iterrows():
        label = row['name']
        conf = row['confidence']
        xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
        color = (0, 255, 0)  # Зеленый цвет для рамки
        cv2.rectangle(filtered_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
        cv2.putText(filtered_image, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    color, 2)

    # Сохраняем обработанное изображение
    processed_file_path = os.path.join(UPLOAD_FOLDER, 'processed_' + file.filename)
    cv2.imwrite(processed_file_path, filtered_image)

    return send_file(processed_file_path, mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(debug=True)

