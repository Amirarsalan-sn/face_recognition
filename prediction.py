from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class_names = {
    0: 'Ali Dayi',
    1: "Mohsen Chavoshi",
    2: 'Mohamad Esfehani',
    3: 'Taraneh Alidosti',
    4: 'Bahram Radan',
    5: 'Sogol Khaligh',
    6: 'Homayoon Shajarian',
    7: 'Sahar Dolatshahi',
    8: 'Mehran Ghafourian',
    9: 'Mehran Modiri',
    10: 'Reza Attaran',
    11: 'Javad Razavian',
    12: 'Seyed Jalal Hoseini',
    13: 'Alireza Beyranvand',
    14: 'Nazanin Bayati',
    15: 'Bahareh Kianafshar',
}

probability_reduction = {
    0: 1,
    1: 1,
    2: 1,
    3: 1,
    4: 1,
    5: 1,
    6: 1,
    7: 1,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 0.82,
    13: 1,
    14: 0.6786,
    15: 1,
}


class MainWindow(QMainWindow):
    def __init__(self, face_models):
        super().__init__()
        self.setWindowTitle("Image face recognizer")
        self.setGeometry(100, 100, 800, 600)
        self.face_models = face_models
        # Create a central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create a vertical layout for the central widget
        layout = QVBoxLayout(central_widget)

        # Create a label for drag and drop section
        self.label = QLabel("Drag and drop an image here", self)
        self.label.setAlignment(Qt.AlignCenter)

        # Create a table for the table section
        self.table = QTableWidget(16, 2, self)
        self.table.setHorizontalHeaderLabels(["Person's name", "Probability"])

        # Add the label and table to the layout
        layout.addWidget(self.label)
        layout.addWidget(self.table)

        # Set the layout for the central widget
        central_widget.setLayout(layout)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            file_path = event.mimeData().urls()[0].toLocalFile()
            y_pred_new = self.process_image(file_path)
            print('received')
            for index in range(len(y_pred_new)):
                item = QTableWidgetItem(f'{y_pred_new[index]}')
                mainWindow.table.setItem(index, 1, item)
            pixmap = QPixmap(file_path)
            self.label.setPixmap(pixmap.scaled(400, 300, Qt.AspectRatioMode.KeepAspectRatio))
            event.accept()

    def process_image(self, image_path):
        # Perform further processing with the image path
        im = Image.open(image_path)
        if im.mode != 'RGB':
            im = im.convert('RGB')

        im = im.resize((200, 200))
        im_copy = np.array(im)
        im_copy = im_copy / 255.0
        im_copy = im_copy.reshape((1, 200, 200, 3))
        y_pred = [0 for i in range(16)]
        for i in range(16):
            y_pred[i] = self.face_models[i].predict(im_copy)[0][0]
            y_pred[i] *= probability_reduction[i]
            if y_pred[i] < 0.01:
                y_pred[i] = 0

        print(y_pred)
        return y_pred


if __name__ == "__main__":
    models = [None for i in range(16)]
    for i in range(16):
        print('loading model: ', i)
        models[i] = load_model(f'D:\\new data set\\saved_models\\saved_model_{i}')

    app = QApplication(sys.argv)
    mainWindow = MainWindow(models)
    for i in range(16):
        item = QTableWidgetItem(class_names[i])
        mainWindow.table.setItem(i, 0, item)
    mainWindow.show()

    sys.exit(app.exec())
