import sys
import time
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication
from tensorflow import keras
import track
import ui

# Список слов для предсказания
words = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'help'
]

def to_numpy_array(img):
    """Преобразует изображение в numpy массив."""
    return np.array(img).reshape(1, 32, 32, 1)

# Инициализация трекера рук и камеры
tracker = track.HandsTrack()
cam_ = cv2.VideoCapture(0)

# Загрузка модели
model = keras.models.load_model('/Users/mishgun/Documents/Python/ML Tensorflow/BaFo/final/ui/model_32.keras')
prediction = None

# Создание экземпляра приложения и главного окна
app = QApplication(sys.argv)
window = ui.MainWindow()
window.show()  # Отображение окна

while True:
    success, img = cam_.read()  # Чтение кадра из камеры
    
    if not success:
        print("Capture not found!")
        break

    # Обновление кадра в окне
    window.update_frame(img)

    # Трекинг рук на изображении
    img = tracker.tracking(img)

    if not tracker.isEmpty:
        prediction = model.predict(to_numpy_array(img))  # Предсказание
        predicted_word = np.argmax(prediction)  # Получение индекса предсказанного слова
        window.set_text(words[predicted_word])  # Установка текста в окно (используем список слов)

    # Выход из цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(0.1)  # Задержка для управления частотой кадров

# Освобождение ресурсов после завершения работы
cam_.release()
cv2.destroyAllWindows()