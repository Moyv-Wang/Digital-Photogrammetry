from PyQt5.QtGui import QPen, QBrush, QColor, QPainter, QPixmap, QTransform
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QCoreApplication, QMetaObject, QRect
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsRectItem, QSizePolicy, QWidget, QTableWidgetItem, QMenuBar, QMenu, QStatusBar, QAction, QTableWidget, \
    QVBoxLayout, QFileDialog


import cv2 as cv
import numpy as np
import math
import csv

pos = None
matches = []
left_img = None
right_img = None
left_gray = None
right_gray = None


class MyGraphicsView(QGraphicsView):
    imageLoaded = pyqtSignal(str)  # 定义图片被加载信号

    def __init__(self):
        super().__init__()
        self._last_pos = None
        self.factor = 1
        self.wheel = True
        self.fileOpened = False
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)  # 设置变换锚点为鼠标位置

    # 更新GraphicsView中的图片
    def wheelEvent(self, event):
        if self.wheel:
            self.factor = 1.2 ** (event.angleDelta().y() / 240)  # 计算缩放因子
            self.scale(self.factor, self.factor)  # 缩放视图
        else:
            super().wheelEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.fileOpened:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._last_pos = event.pos()
        elif event.button() == Qt.RightButton:
            # 弹出菜单
            menu = QMenu(self)
            add = QAction('添加', self)
            menu.addAction(add)
            # 点击添加后，弹出文件选择对话框
            add.triggered.connect(self.openfile)
            menu.exec_(event.globalPos())
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton and self.fileOpened:
            delta = event.pos() - self._last_pos
            self._last_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.setDragMode(QGraphicsView.NoDrag)
        super(MyGraphicsView, self).mouseReleaseEvent(event)

    def openfile(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Image Files (*.png *.jpg *.bmp)")
        if fileName:
            self.fileOpened = True
            self.imageLoaded.emit(fileName)  # 发送信号
            self.scene = QGraphicsScene()
            self.scene.addPixmap(QPixmap(fileName))
            self.setScene(self.scene)
            self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.show()