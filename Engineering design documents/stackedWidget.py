import  sys,os
sys.path.append('..')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    #先加入绝对路径，否则会报错，注意__file__表示的是当前执行文件的路径

from custom.tableWidget import *
from config import tables


class StackedWidget(QStackedWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        for table in tables:
            self.addWidget(table(parent=parent))
        self.setMinimumWidth(300)
