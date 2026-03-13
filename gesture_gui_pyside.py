import sys
from PySide2.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QProgressBar, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PySide2.QtCore import Qt, QTimer
from PySide2.QtGui import QColor, QPalette


class GestureGUI(QWidget):
    """
    PySide2 手勢辨識 GUI，顯示 4 種手勢的機率條狀圖，並突顯當前辨識結果。
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Recognition")
        self.resize(600, 400)

        # **主要 Layout**
        main_layout = QVBoxLayout()

        # **當前手勢標籤**
        self.current_gesture_label = QLabel("Current gesture: Background")
        self.current_gesture_label.setAlignment(Qt.AlignCenter)
        self.current_gesture_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; padding: 10px; background-color: lightgray; border-radius: 5px;"
        )
        main_layout.addWidget(self.current_gesture_label)

        # **進度條區域**
        self.hbox = QHBoxLayout()

        # **手勢名稱 & 條狀圖**
        self.gesture_names = ["Background", "PatPat", "Wave", "Come"]
        self.bars = {}  # 存放進度條物件

        # **可調整的參數**
        self.BAR_WIDTH = 15  # 進度條寬度
        self.bar_colors = {
            "Background": "green",
            "PatPat": "blue",
            "Wave": "red",
            "Come": "purple"
        }
        self.gesture_colors = {
            "Background": "lightgray",
            "PatPat": "#ADD8E6",
            "Wave": "#FFCCCB",
            "Come": "#E6E6FA"
        }

        # **Spacer 讓條狀圖置中**
        self.hbox.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        for name in self.gesture_names:
            # 建立一個垂直 Layout
            v_layout = QVBoxLayout()

            # 進度條
            bar = QProgressBar()
            bar.setOrientation(Qt.Vertical)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)  # 不顯示文字
            bar.setStyleSheet(f"QProgressBar::chunk {{ background-color: {self.bar_colors[name]}; }}")
            bar.setFixedWidth(self.BAR_WIDTH)  # 設定進度條寬度
            v_layout.addWidget(bar, alignment=Qt.AlignBottom)

            # 手勢標籤
            label = QLabel(name)
            label.setAlignment(Qt.AlignCenter)
            v_layout.addWidget(label, alignment=Qt.AlignCenter)

            self.hbox.addLayout(v_layout)

            # 在每個進度條之間加入 SpacerItem，讓它們等距排列
            self.hbox.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

            self.bars[name] = bar  # 存入字典

        # **Spacer 讓條狀圖置中**
        self.hbox.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum))

        main_layout.addLayout(self.hbox)
        self.setLayout(main_layout)

    def update_probabilities(self, background_prob, patpat_prob, wavewave_prob, come_prob, current_gesture):
        """
        更新 4 個手勢的機率進度條與辨識結果。

        :param background_prob: float, 背景機率 [0,1]
        :param patpat_prob: float, PatPat 手勢機率 [0,1]
        :param wavewave_prob: float, WaveWave 手勢機率 [0,1]
        :param come_prob: float, Come 手勢機率 [0,1]
        :param current_gesture: str, 當前辨識出的手勢名稱
        """
        # 轉換為百分比
        self.bars["Background"].setValue(int(background_prob * 100))
        self.bars["PatPat"].setValue(int(patpat_prob * 100))
        self.bars["Wave"].setValue(int(wavewave_prob * 100))
        self.bars["Come"].setValue(int(come_prob * 100))

        # 更新中央標籤
        self.current_gesture_label.setText(f"Current gesture: {current_gesture}")
        self.current_gesture_label.setStyleSheet(
            f"font-size: 20px; font-weight: bold; padding: 10px; background-color: {self.gesture_colors[current_gesture]}; border-radius: 5px;"
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GestureGUI()
    window.show()

    # 測試數據：每秒更新一次
    import random

    def simulate_data():
        bg = random.uniform(0, 1)
        pp = random.uniform(0, 1 - bg)
        ww = random.uniform(0, 1 - bg - pp)
        cm = 1 - (bg + pp + ww)

        if pp > 0.5:
            gesture = "PatPat"
        elif ww > 0.5:
            gesture = "Wave"
        elif cm > 0.5:
            gesture = "Come"
        else:
            gesture = "Background"

        window.update_probabilities(bg, pp, ww, cm, gesture)

    timer = QTimer()
    timer.timeout.connect(simulate_data)
    timer.start(1000)  # 每 1000 毫秒更新一次

    sys.exit(app.exec_())
