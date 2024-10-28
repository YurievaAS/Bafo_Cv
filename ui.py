import sys
import cv2
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QLabel, QVBoxLayout, QLineEdit
from PyQt6.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    #конструктор
    def __init__(self):
        super(MainWindow, self).__init__() #вызывает конструктор базового класса

        self.setWindowTitle("Sign Language recognition")
        self.capture = cv2.VideoCapture(0)
        if  not self.capture.isOpened():  # Проверка, была ли веб-камера открыта
            print("не удалось открыть веб-камеру")
            sys.exit()
        # Создаем QLabel для отображения видео
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Создаем вертикальный layout
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(24)  # Обновление каждые 24 мс (примерно 24 кадра в секунду)

    def update_frame(self):
        success, frame = self.capture.read()
        if success:
            print("удалось открыть веб-камеру")
            # Преобразование изображения в формат QImage
            image = QImage(frame, frame.shape[1], frame.shape[0],
                           QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)
        else:
            print("Ошибка получения кадра")

    def closeEvent(self, event):
        # Выключение веб-камеры при закрытии приложения
        self.capture.release()
        super().closeEvent(event)

app = QApplication(sys.argv)
window = MainWindow()
window.show()
#запустить цикл событий
sys.exit(app.exec())


''''
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Не удалось открыть веб-камеру")
            sys.exit()
        '''