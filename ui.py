import sys
import cv2
from PyQt5.QtCore import QSize, Qt, QTimer
from PyQt5.QtWidgets import \
    QApplication, QMainWindow, QPushButton, \
    QWidget, QLabel,QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QImage, QPixmap

class MainWindow(QMainWindow):
    #конструктор
    def __init__(self):
        super(MainWindow, self).__init__() #вызывает конструктор базового класса
        self.setWindowTitle("Sign Language recognition")
        self.setFixedSize(QSize(1000,800))

        self.clear_input_button = QPushButton('Clear')
        self.clear_input_button.setStyleSheet("background-color:pink")
        self.clear_input_button.clicked.connect(lambda  : self.clear_text())

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout_1 = QHBoxLayout()

        self.video = QLabel()
        self.video.setStyleSheet("background-color:pink")

        self.input = QLabel()
        self.input.setStyleSheet("background-color:lightgray")
        self.text = ''


        layout.addWidget(self.video, stretch=2)
        layout.addLayout(layout_1, stretch=1)
        layout_1.addWidget(self.input, stretch=6)
        layout_1.addWidget(self.clear_input_button, stretch=1)
        #layout.addWidget(self.input, stretch=1)

        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(24)  # Обновление каждые 24 мс (примерно 24 кадра в секунду)

        self.cap = cv2.VideoCapture(0)

    def update_frame(self):
        width = self.video.width()
        height = self.video.height()
        success, img = self.cap.read()
        img = cv2.resize(img, (width,height))
        if success:
            image = QImage(img, img.shape[1], img.shape[0],
                               QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(image)
            self.video.setPixmap(pixmap)
        else:
            print("Ошибка получения кадра")

    def set_text(self, text):
        self.text += text
        self.input.setText(self.text)

    def clear_text(self):
        self.input.setText('')




app = QApplication(sys.argv)
window = MainWindow()
window.set_text('aaaa')
window.show()
sys.exit(app.exec())


