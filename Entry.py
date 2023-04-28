import sys
import csv
import cv2 as cv
import numpy as np
import math
import os
import pandas as pd

from PyQt5.QtWidgets import QDialog, QSizePolicy, QLabel, QWidget, QHBoxLayout, QLineEdit, QPushButton, QMainWindow, \
    QFileDialog, QApplication, QTableWidgetItem, QGraphicsScene, QInputDialog, QMessageBox
from PyQt5.QtCore import pyqtSignal, QRect, Qt, QObject, QPoint
from PyQt5.QtGui import QPixmap, QStandardItemModel, QStandardItem
import MainWidget
import utils
import Correlation_dialog
import LST_dialog


class MyWidget(QWidget):
    img_src = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self.harris_kernel_window_size = 0
        self.harris_search_window_size = 0
        self.myUtil = utils.Utils()
        self.dialog = Correlation_dialog.MyDialog()
        self.dialog2 = LST_dialog.MyDialog()
        self.ui = MainWidget.Ui_Digital_Photogrammetry()
        self.model = QStandardItemModel()
        self.message = QMessageBox()
        self.scaled = False
        self.ui.setupUi(self)
        self.ui.harris_btn.clicked.connect(lambda: utils.Utils.detectHarrisCorner(self.myUtil, self.img_src[0]))
        self.myUtil.left_imageProcessed.connect(self.harris_finished)
        self.ui.correlation_btn.clicked.connect(self.harris_dia)
        self.ui.lst_btn.clicked.connect(self.lst_dia)
        # self.ui.correlation_btn.clicked.connect(
        #     lambda: utils.Utils.Harris(self.myUtil, self.img_src[1],kernel_window_size=9, search_window_size=11))
        self.myUtil.right_imageProcessed.connect(self.correlation_finished)
        # self.ui.lst_btn.clicked.connect(
        #     lambda: utils.Utils.ordinary_least_square(self.myUtil, window_size=11))
        self.myUtil.LST_Processed.connect(self.lst_finished)
        # self.ui.export_btn.clicked.connect(lambda: utils.)
        # 连接imageLoaded信号来接收左片的路径
        self.ui.gv_left.imageLoaded.connect(self.set_img)
        # 连接imageLoaded信号来接收右片的路径
        self.ui.gv_right.imageLoaded.connect(self.set_img)
        self.ui.export_btn.clicked.connect(self.export_finished)
        self.ui.tb.clicked.connect(self.tb_clicked)

    def tb_clicked(self, index):
        # 获取点击行的同名点坐标
        l_x = int(self.ui.tb.model().index(index.row(), 0).data())
        l_y = int(self.ui.tb.model().index(index.row(), 1).data())
        r_x = int(self.ui.tb.model().index(index.row(), 2).data())
        r_y = int(self.ui.tb.model().index(index.row(), 3).data())
        l_point = QPoint(l_x, l_y)
        l_center = self.ui.gv_left.mapFromScene(self.ui.gv_left.mapToScene(l_point))
        self.ui.gv_left.centerOn(l_center)
        r_point = QPoint(r_x, r_y)
        r_center = self.ui.gv_right.mapFromScene(self.ui.gv_right.mapToScene(r_point))
        self.ui.gv_right.centerOn(r_center)
        if not self.scaled:
            self.ui.gv_left.scale(8, 8)
            self.ui.gv_right.scale(8, 8)
            self.scaled = True
    
    def harris_finished(self):
        self.updateGraphicsView(self.ui.gv_left, "./left_match.jpg")
        self.message.setText("左片Harris角点检测完成!")
        self.message.exec_()

    def correlation_finished(self):
        self.updateGraphicsView(self.ui.gv_left, "./left_match.jpg")
        self.updateGraphicsView(self.ui.gv_right, "./right_match.jpg")
        data = pd.read_csv('Matches.csv')
        self.model.setColumnCount(len(data.columns))
        self.model.setRowCount(len(data.index))
        for row in range(len(data.index)):
            for col in range(len(data.columns)):
                item = QStandardItem(str(data.iloc[row, col]))
                self.model.setItem(row, col, item)
        self.ui.tb.setModel(self.model)
        self.ui.gv_left.wheel = False
        self.ui.gv_right.wheel = False
        self.message.setText("右片相关系数匹配完成!\n点击某一行即可缩放至对应同名点\n你会发现滚轮没用了\n不用怀疑，现在我把它禁用了")
        self.message.exec_()

    def lst_finished(self):
        data = pd.read_csv('LST_Matches.csv')
        self.model.removeRows(0, self.model.rowCount())
        self.model.setColumnCount(len(data.columns))
        self.model.setRowCount(len(data.index))
        for row in range(len(data.index)):
            for col in range(len(data.columns)):
                item = QStandardItem(str(data.iloc[row, col]))
                self.model.setItem(row, col, item)
        self.ui.tb.setModel(self.model)
        self.message.setText("最小二乘匹配完成!")
        self.message.exec_()

    def export_finished(self):
        self.message.setText("相关系数匹配结果导出为Matches.csv\n最小二乘匹配结果导出为LST_Matches.csv")
        self.message.exec_()

    def harris_dia(self):
        if self.dialog.exec_() == QDialog.Accepted:
            value1, value2 = self.dialog.getValues()
            if value1 != '' and value2 != '':
                utils.Utils.Harris(self.myUtil, self.img_src[1], int(value1), int(value2))
            if value1 == '' and value2 != '':
                utils.Utils.Harris(self.myUtil, self.img_src[1], 9, int(value2))
            if value2 == '' and value1 != '':
                utils.Utils.Harris(self.myUtil, self.img_src[1], int(value1), 11)
            if value1 == '' and value2 == '':
                utils.Utils.Harris(self.myUtil, self.img_src[1], 9, 11)

    def lst_dia(self):
        if self.dialog2.exec_() == QDialog.Accepted:
            value = self.dialog2.getValues()
            if value != '':
                utils.Utils.ordinary_least_square(self.myUtil, int(value))
            else:
                utils.Utils.ordinary_least_square(self.myUtil, 11)

    def updateGraphicsView(self, object, str):
        object.scene.clear()
        object.scene = QGraphicsScene()
        print(os.path.exists(str))
        object.scene.addPixmap(QPixmap(str))
        object.setScene(object.scene)
        object.fitInView(object.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
        object.show()

    # 接收原图的路径
    def set_img(self, img):
        self.img_src.append(img)
        print(self.img_src[0])


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MyWidget()
    w.show()
    app.exec()
