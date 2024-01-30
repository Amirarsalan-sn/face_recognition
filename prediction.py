from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class_names = {
    0: 'Ali Day',
    1: "Mohsen Chavoshi",
    2: 'Mohamad Esfehani',
    3: 'Taraneh Alidostnia',
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


class MainWindow(QMainWindow):
    def __init__(self, face_model):
        super().__init__()
        self.setWindowTitle("Image face recognizer")
        self.setGeometry(100, 100, 800, 600)
        self.face_model = face_model
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
        im = im.resize((100, 100))
        im_copy = np.array(im)
        if im_copy.shape != (100, 100, 3):
            expanded_image_array = np.expand_dims(im_copy, axis=2)
            im_copy = np.repeat(expanded_image_array, 3, axis=2)

        im_copy = im_copy / 255.0
        y_pred = self.face_model.predict(im_copy.reshape(1, 100, 100, 3))
        y_pred_new = [0 if x < 0.01 else x for x in y_pred[0]]
        print(y_pred_new)
        return y_pred_new


if __name__ == "__main__":
    model = load_model('saved_model')

    app = QApplication(sys.argv)
    mainWindow = MainWindow(model)
    for person in class_names.keys():
        item = QTableWidgetItem(class_names[person])
        mainWindow.table.setItem(person, 0, item)
    mainWindow.show()

    sys.exit(app.exec())
