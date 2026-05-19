import sys
import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import matplotlib.pyplot as plt

from custom.stackedWidget import StackedWidget
from custom.treeView import FileSystemTreeView
from custom.listWidgets import FuncListWidget, UsedListWidget
from custom.graphicsView import GraphicsView


class MyApp(QMainWindow):
    def __init__(self):
        super(MyApp, self).__init__()

        # # ================= 工具栏 =================
        # self.tool_bar = self.addToolBar('工具栏')
        # self.action_right_rotate = QAction(QIcon("icons/right.png"), "向右旋转90", self)
        # self.action_left_rotate = QAction(QIcon("icons/left.png"), "向左旋转90°", self)
        #
        # self.action_right_rotate.triggered.connect(self.right_rotate)
        # self.action_left_rotate.triggered.connect(self.left_rotate)
        #
        # self.tool_bar.addActions((self.action_left_rotate, self.action_right_rotate))
        # ================= 工具栏 =================
        self.tool_bar = QToolBar("工具栏", self)
        self.tool_bar.setIconSize(QSize(28, 28))  # 图标大小（关键）
        self.tool_bar.setMovable(False)  # 禁止拖动（更像产品）
        self.tool_bar.setFloatable(False)

        self.addToolBar(Qt.TopToolBarArea, self.tool_bar)
        self.tool_bar.setObjectName("mainToolBar")
        # 添加按钮
        self.action_left_rotate = QAction(QIcon("icons/left.png"), "左旋转", self)
        self.action_right_rotate = QAction(QIcon("icons/right.png"), "右旋转", self)

        self.action_left_rotate.triggered.connect(self.left_rotate)
        self.action_right_rotate.triggered.connect(self.right_rotate)

        # 添加到工具栏
        self.tool_bar.addAction(self.action_left_rotate)
        self.tool_bar.addAction(self.action_right_rotate)

        # 👉 加分割（更清晰）
        self.tool_bar.addSeparator()
        # ================= 核心组件 =================
        self.useListWidget = UsedListWidget(self)
        self.funcListWidget = FuncListWidget(self)
        self.stackedWidget = StackedWidget(self)
        self.fileSystemTreeView = FileSystemTreeView(self)

        # ⭐ 两个图像视图（核心改动）
        self.graphicsView_src = GraphicsView(self)  # 原图
        self.graphicsView_dst = GraphicsView(self)  # 处理后

        # ================= 中央对比区域 =================
        self.init_center_widget()

        # ================= Dock区域 =================
        self.init_dock_widgets()

        # ================= 窗口设置 =================
        self.setWindowTitle('"医影智元—医学影像智能处理平台')
        self.setWindowIcon(QIcon('icons/TZB.png'))

        # ================= 图像数据 =================
        self.src_img = None
        self.cur_img = None

    # =====================================================
    # ⭐ 中央区域（双窗口）
    # =====================================================
    def init_center_widget(self):
        # 左侧：处理前
        left_layout = QVBoxLayout()
        left_label = QLabel("处理前")
        left_label.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(left_label)
        left_layout.addWidget(self.graphicsView_src)

        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        # 右侧：处理后
        right_layout = QVBoxLayout()
        right_label = QLabel("处理后")
        right_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(right_label)
        right_layout.addWidget(self.graphicsView_dst)

        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        # ⭐ 分割器（可拖动）
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(left_widget)
        self.splitter.addWidget(right_widget)
        self.splitter.setSizes([1, 1])  # 初始比例

        self.setCentralWidget(self.splitter)

    # =====================================================
    # Dock初始化
    # =====================================================
    def init_dock_widgets(self):
        # 文件
        self.dock_file = QDockWidget(self)
        self.dock_file.setWidget(self.fileSystemTreeView)
        self.dock_file.setTitleBarWidget(QLabel('选择文件'))
        self.dock_file.setFeatures(QDockWidget.NoDockWidgetFeatures)

        # 功能
        self.dock_func = QDockWidget(self)
        self.dock_func.setWidget(self.funcListWidget)
        self.dock_func.setTitleBarWidget(QLabel('图像操作'))
        self.dock_func.setFeatures(QDockWidget.NoDockWidgetFeatures)

        # 已用操作
        self.dock_used = QDockWidget(self)
        self.dock_used.setWidget(self.useListWidget)
        self.dock_used.setTitleBarWidget(QLabel('当前操作'))
        self.dock_used.setFeatures(QDockWidget.NoDockWidgetFeatures)

        # 属性
        self.dock_attr = QDockWidget(self)
        self.dock_attr.setWidget(self.stackedWidget)
        self.dock_attr.setTitleBarWidget(QLabel('属性'))
        self.dock_attr.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.dock_attr.close()

        # 布局
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_file)
        self.addDockWidget(Qt.TopDockWidgetArea, self.dock_func)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_used)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_attr)

    # =====================================================
    # 图像处理逻辑
    # =====================================================
    def process_image(self):
        img = self.src_img.copy()
        for i in range(self.useListWidget.count()):
            img = self.useListWidget.item(i)(img)
        return img

    # =====================================================
    # 更新图像（只更新右侧）
    # =====================================================
    def update_image(self):
        if self.src_img is None:
            return

        img = self.process_image()
        self.cur_img = img
        self.graphicsView_dst.update_image(img)

    # =====================================================
    # 更换图像（左右同步）
    # =====================================================
    def change_image(self, img):
        self.src_img = img

        # 左：原图
        self.graphicsView_src.change_image(img)

        # 右：处理后
        img = self.process_image()
        self.cur_img = img
        self.graphicsView_dst.change_image(img)

    # =====================================================
    # 旋转（同步）
    # =====================================================
    def right_rotate(self):
        self.graphicsView_src.rotate(90)
        self.graphicsView_dst.rotate(90)

    def left_rotate(self):
        self.graphicsView_src.rotate(-90)
        self.graphicsView_dst.rotate(-90)

    # =====================================================
    # 直方图
    # =====================================================
    def histogram(self):
        if self.cur_img is None:
            return

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([self.cur_img], [i], None, [256], [0, 256])
            histr = histr.flatten()
            plt.plot(range(256), histr, color=col)
            plt.xlim([0, 256])

        plt.show()


# =====================================================
# 主程序入口
# =====================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # QSS样式
    app.setStyleSheet(open('./custom/styleSheet.qss', encoding='utf-8').read())

    window = MyApp()
    window.setWindowIcon(QIcon('icons/main.png'))
    window.show()

    sys.exit(app.exec_())