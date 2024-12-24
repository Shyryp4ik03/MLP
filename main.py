import pandas as pd
import numpy as np

from sklearn import linear_model, ensemble, metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from PIL import ImageTk, Image, ImageDraw
import PIL
from tkinter import *
import datetime
import os

from matplotlib import pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, activations, losses, optimizers, metrics

# Настройка модели TensorFlow
with tf.device('/CPU:0'):
    input_size = 1
    hiddenLayer_size = 3
    output_size = 1

    model = tf.keras.models.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(units=hiddenLayer_size, activation=None),
        layers.Activation(activation=activations.relu),
        layers.Dense(units=output_size, activation=None)
    ])

with tf.device('/CPU:0'):
    fLoss = losses.CategoricalCrossentropy()
    fOptimizer = optimizers.Adam(learning_rate=0.01)
    fMetric = [
        metrics.CategoricalAccuracy(),
        metrics.CategoricalCrossentropy(),
        metrics.Precision()
    ]

    model.compile(
        loss=fLoss,
        optimizer=fOptimizer,
        metrics=fMetric
    )

def initialize_layer_weights(layer, weights, biases):
    layer.set_weights([np.array(weights), np.array(biases)])

initialize_layer_weights(model.layers[0], [[0.5, 0.4, 1.0]], [0.3, 0.6, 0.9])
initialize_layer_weights(model.layers[2], [[2.0], [3.0], [5.0]], [0.75])

# Пример прогноза
x = np.array([[1.0]])
print("Model Prediction for x = [1.0]:", model.predict(x))

# Приложение с графическим интерфейсом для рисования
class MyWinTK():
    def saveToFileFunc(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f'img_{timestamp}.bmp'
        saved_image = self.output_image.resize((self.widthSave, self.heightSave), PIL.Image.BILINEAR)
        saved_image.save(os.path.join(self.pathFotImgs, filename), quality=100, subsampling=0)

    def drawGrid(self):
        for ih in range(0, self.widthGUI // self.gridSize):
            self.canvasTk.create_line(0, ih * self.gridSize, self.widthGUI, ih * self.gridSize, fill='#555555')
        for iw in range(0, self.heightGUI // self.gridSize):
            self.canvasTk.create_line(iw * self.gridSize, 0, iw * self.gridSize, self.heightGUI, fill='#555555')

    def clearFunc(self):
        self.draw.rectangle((0, 0, self.widthGUI, self.heightGUI), fill=(0, 0, 0, 0))
        self.canvasTk.delete("all")
        self.canvasTk.create_rectangle(0, 0, self.widthGUI, self.heightGUI, fill='black')
        self.drawGrid()

    def paintFunc(self, event):
        grx = event.x // self.gridSize * self.gridSize
        gry = event.y // self.gridSize * self.gridSize
        x1, y1 = grx, gry
        x2, y2 = grx + self.gridSize, gry + self.gridSize
        self.canvasTk.create_rectangle(x1, y1, x2, y2, fill="white", width=1)
        self.draw.rectangle((x1, y1, x2, y2), fill=(255, 255, 255, 0))

    def __init__(self):
        self.widthGUI, self.heightGUI = 200, 200
        self.widthSave, self.heightSave = 20, 20
        self.gridSize = 10
        self.pathFotImgs = 'imgs'

        if not os.path.exists(self.pathFotImgs):
            os.mkdir(self.pathFotImgs)

        rootWindow = Tk()
        self.canvasTk = Canvas(rootWindow, width=self.widthGUI, height=self.heightGUI, bg='black')
        self.canvasTk.bind("<B1-Motion>", self.paintFunc)
        self.canvasTk.bind("<Button-1>", self.paintFunc)
        self.output_image = PIL.Image.new("RGB", (self.widthGUI, self.heightGUI), (0, 0, 0))
        self.draw = ImageDraw.Draw(self.output_image)
        self.drawGrid()
        self.canvasTk.pack(expand=YES, fill=BOTH)

        Button(text="Сохранить в файл", command=self.saveToFileFunc, bg='#CCFFCC').pack()
        Button(text="Очистить", command=self.clearFunc, bg='#FFCCCC').pack()

        Label(rootWindow, text="После рисования и сохранения\n нужного кол-ва изображений\n обязательно закройте это окно").pack()

        rootWindow.title("Рисование изображений")
        rootWindow.mainloop()


if __name__ == "__main__":
    print("Starting TensorFlow prediction and GUI application...")
    MyWinTK()