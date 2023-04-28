from PyQt5.QtWidgets import QDialog, QLabel, QLineEdit, QDialogButtonBox, QVBoxLayout

class MyDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Operator Size")
        self.result = None

        # 第一个文本框
        self.label1 = QLabel("输入目标窗口大小（需为奇数）:")
        self.textbox1 = QLineEdit(self)
        self.textbox1.setPlaceholderText("默认为9")
        self.textbox1.returnPressed.connect(self.accept)

        # 第二个文本框
        self.label2 = QLabel("输入搜索窗口大小（需为奇数）:")
        self.textbox2 = QLineEdit(self)
        self.textbox2.setPlaceholderText("默认为11")
        self.textbox2.returnPressed.connect(self.accept)

        # 确定和取消按钮
        self.buttonbox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonbox.accepted.connect(self.accept)
        self.buttonbox.rejected.connect(self.reject)

        # 设置布局
        layout = QVBoxLayout()
        layout.addWidget(self.label1)
        layout.addWidget(self.textbox1)
        layout.addWidget(self.label2)
        layout.addWidget(self.textbox2)
        layout.addWidget(self.buttonbox)
        self.setLayout(layout)

    def getValues(self):
        return self.textbox1.text(), self.textbox2.text()
