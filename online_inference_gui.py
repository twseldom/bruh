# realtime_infer_with_gui.py
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PySide2 import QtWidgets
import threading
import serial

# ======== 路徑/參數（可直接改） ========
MODEL_PATH   = r"D:\mmWave\radar-gesture-recognition-chore-update-20250815\src\3d_cnn_model.pth"
SETTING_FILE = r"D:\mmWave\radar-gesture-recognition-chore-update-20250815\TempParam\K60168-Test-00256-008-v0.0.8-20230717_60cm"

WINDOW_SIZE  = 30                    # 滑動視窗幀數
CLASS_NAMES  = ["Background", "PatPat", "Wave", "Come"]  # 你的輸出順序
ENTER_TH     = 0.40                  # 進入閥值（高）
EXIT_TH      = 0.20                  # 退出閥值（低）
STREAM_TYPE  = "feature_map"         # 或 "raw_data"
# ======================================

# ======== 你的 GUI 元件 ========
# 需提供 gesture_gui_pyside.py，且類別有 update_probabilities(bg, pat, wave, come, current)
from gesture_gui_pyside import GestureGUI

# ======== Kaiku / KKT imports ========
from KKT_Module import kgl
from KKT_Module.DataReceive.Core import Results
from KKT_Module.DataReceive.DataReceiver import MultiResult4168BReceiver
from KKT_Module.FiniteReceiverMachine import FRM
from KKT_Module.SettingProcess.SettingConfig import SettingConfigs
from KKT_Module.SettingProcess.SettingProccess import SettingProc
from KKT_Module.GuiUpdater.GuiUpdater import Updater

# ---------- Kaiku helpers ----------
def connect_device():
    try:
        device = kgl.ksoclib.connectDevice()
        if device == 'Unknow':
            ret = QtWidgets.QMessageBox.warning(
                None, 'Unknown Device', 'Please reconnect device and try again',
                QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
            )
            if ret == QtWidgets.QMessageBox.Ok:
                connect_device()
    except Exception:
        ret = QtWidgets.QMessageBox.warning(
            None, 'Connection Failed', 'Please reconnect device and try again',
            QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel
        )
        if ret == QtWidgets.QMessageBox.Ok:
            connect_device()

def run_setting_script(setting_name: str):
    ksp = SettingProc()
    cfg = SettingConfigs()
    cfg.Chip_ID = kgl.ksoclib.getChipID().split(' ')[0]
    cfg.Processes = [
        'Reset Device',
        'Gen Process Script',
        'Gen Param Dict', 'Get Gesture Dict',
        'Set Script',
        'Run SIC',
        'Phase Calibration',
        'Modulation On'
    ]
    cfg.setScriptDir(f'{setting_name}')
    ksp.startUp(cfg)

def set_properties(obj: object, **kwargs):
    print(f"==== Set properties in {obj.__class__.__name__} ====")
    for k, v in kwargs.items():
        if not hasattr(obj, k):
            print(f'Attribute "{k}" not in {obj.__class__.__name__}.')
            continue
        setattr(obj, k, v)
        print(f'Attribute "{k}", set "{v}"')

# ---------- 3D CNN（與訓練一致的 classifier.* 命名） ----------
class Gesture3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv3d(2, 32, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(64),
            nn.Conv3d(64,128, 3), nn.ReLU(), nn.MaxPool3d(2), nn.BatchNorm3d(128),
        )
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.classifier = nn.Sequential(
            nn.Linear(128,128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)   # logits

def _maybe_remap_keys_to_classifier(state: dict) -> dict:
    # 若權重鍵名是 fc.*，轉成 classifier.*（保險）
    if any(k.startswith("fc.") for k in state.keys()):
        new = {}
        for k, v in state.items():
            new["classifier." + k[3:]] = v if k.startswith("fc.") else v
        return new
    return state

# ---------- 即時推論核心 ----------
class OnlineInferenceContext:
    def __init__(self, model: nn.Module, device: torch.device, window_size: int):
        self.model = model
        self.device = device
        self.window = window_size
        self.buffer = np.zeros((2, 32, 32, self.window), dtype=np.float32)
        self.collected = 0
        # 雙閥值狀態
        self.active = False
        self.last_pred = "Background"

    @staticmethod
    def to_frame(arr) -> np.ndarray:
        x = np.asarray(arr)
        # 常見兩種： (2,32,32) 或 (32,32,2)
        if x.shape == (2, 32, 32):
            pass
        elif x.shape == (32, 32, 2):
            x = np.transpose(x, (2, 0, 1))
        else:
            raise ValueError(f"Unexpected frame shape: {x.shape}")
        return x.astype(np.float32, copy=True)

    def push_and_infer(self, frame: np.ndarray):
        # 滑動與塞入
        self.buffer = np.roll(self.buffer, shift=-1, axis=-1)
        self.buffer[..., -1] = frame
        self.collected += 1
        if self.collected < self.window:
            return None  # 還沒滿，先不推論

        # (2,32,32,T) -> (1,2,T,32,32)
        win = np.expand_dims(self.buffer, axis=0)
        win = np.transpose(win, (0, 1, 4, 2, 3))
        x = torch.from_numpy(win).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x)          # (1,4)
            p = F.softmax(logits, dim=1).cpu().numpy()[0]  # [BG, PatPat, Wave, Come]
        return p

    def apply_double_threshold(self, probs: np.ndarray):
        bg, come, pat, wave = probs
        nonbg = np.array([pat, wave, come])
        nonbg_names = CLASS_NAMES[1:]  # ["PatPat","Wave","Come"]
        top_idx = int(nonbg.argmax())
        top_name = nonbg_names[top_idx]
        top_prob = float(nonbg[top_idx])

        if not self.active:
            if top_prob >= ENTER_TH:
                self.active = True
                current = top_name
            else:
                current = "Background"
        else:
            if (nonbg < EXIT_TH).all():
                self.active = False
                current = "Background"
            else:
                current = self.last_pred  # 鎖定在上一個手勢直到掉回 EXIT_TH 下

        changed = (current != self.last_pred)
        self.last_pred = current
        return current, changed, (bg, pat, wave, come)

# ---------- Updater：把 Results 餵進推論 + 更新 GUI ----------
class InferenceUpdater(Updater):
    def __init__(self, ctx: OnlineInferenceContext, gesture_gui: GestureGUI,
                 stream: str = "feature_map",
                 arduino_ser: serial.Serial = None):
        super().__init__()
        self.ctx = ctx
        self.gui = gesture_gui
        self.stream = stream

        self.arduino = arduino_ser     # UNO 串口物件
        self._last_cmd = None          # 上一次送的指令（避免狂刷）
        self._lock = threading.Lock()
        self.floor = 1                 # 目前樓層（1 or 2）

    def _gesture_to_cmd_and_floor(self, current: str):
        """
        CLASS_NAMES = ["Background", "PatPat", "Wave", "Come"]

        規則：
        - PatPat / Come = 上樓（U → 2 樓）
        - Wave          = 下樓（D → 1 樓）
        - Background    = 不送指令（None）
        """

        if current in ("PatPat", "Come"):
            self.floor = 2
            cmd = "U"
        elif current == "Wave":
            self.floor = 1
            cmd = "D"
        else:
            cmd = None

        print(f"[MAP] gesture={current} -> cmd={cmd}, floor={self.floor}")
        return cmd

    def _send_to_arduino(self, cmd: str):
        # 不送的情況先擋掉
        if cmd is None:
            print("[UNO] skip send (cmd=None)")
            return
        if self.arduino is None:
            print("[UNO] arduino_ser is None，沒連上")
            return
        if cmd == self._last_cmd:
            # 同一個狀態連續出現就不重複送
            print(f"[UNO] same cmd '{cmd}' as last, skip")
            return

        with self._lock:
            try:
                self.arduino.write(cmd.encode("ascii"))
                self._last_cmd = cmd
                print(f"[UNO] send cmd={cmd} to Arduino")
            except Exception as e:
                print("[UNO] write failed:", e)

    def update(self, res: Results):
        try:
            if self.stream == "raw_data":
                arr = res['raw_data'].data
            else:
                arr = res['feature_map'].data

            frame = self.ctx.to_frame(arr)
            probs = self.ctx.push_and_infer(frame)
            if probs is None:
                return

            current, changed, (bg, pat, wave, come) = self.ctx.apply_double_threshold(probs)

            # 更新 GUI 顯示
            try:
                self.gui.update_probabilities(
                    float(bg), float(pat), float(wave), float(come), current
                )
            except Exception:
                pass

            # 只有「手勢狀態改變」時才送指令
            if changed:
                print(f"[Pred] current={current} | BG:{bg:.2f} Pat:{pat:.2f} Wave:{wave:.2f} Come:{come:.2f}")
                cmd = self._gesture_to_cmd_and_floor(current)
                self._send_to_arduino(cmd)

        except Exception as e:
            print("[Updater] exception:", e)
            # 不 raise，避免整個收到卡死

# ---------- 主流程 ----------
def main():
    # 0) Qt 事件圈
    app = QtWidgets.QApplication(sys.argv)

    # 1) 啟動你的 GUI
    gui = GestureGUI()
    gui.show()

    # 2) 初始化雷達
    kgl.setLib()
    connect_device()
    run_setting_script(SETTING_FILE)

    # 切換輸出源（與你 GUI 版相同的寄存器設定）
    if STREAM_TYPE == "raw_data":
        kgl.ksoclib.writeReg(0, 0x50000504, 5, 5, 0)
    else:
        kgl.ksoclib.writeReg(1, 0x50000504, 5, 5, 0)

    # 3) 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Gesture3DCNN(num_classes=4).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = _maybe_remap_keys_to_classifier(state)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[INFO] model loaded: {MODEL_PATH}  | device: {device}")

    # 4) 上線推論（帶入 GUI）
    ctx = OnlineInferenceContext(model=model, device=device, window_size=WINDOW_SIZE)
    # ★ Arduino 串口：用跟 uno_test.py 一樣的 COM 號 ★
    try:
        ARDUINO_PORT = "COM4"   # ←←← 改成你實際的，比如 "COM6"
        arduino_ser = serial.Serial(ARDUINO_PORT, 9600, timeout=0)
        time.sleep(2)
        print(f"[INFO] Arduino connected on {ARDUINO_PORT}")
    except Exception as e:
        print("[ERROR] 無法連接 Arduino：", e)
        arduino_ser = None
    updater = InferenceUpdater(
        ctx,
        gesture_gui=gui,
        stream=STREAM_TYPE,
        arduino_ser=arduino_ser,
    )

    # 5) Receiver + FRM
    receiver = MultiResult4168BReceiver()
    set_properties(receiver,
                   actions=1,
                   rbank_ch_enable=7,
                   read_interrupt=0,
                   clear_interrupt=0)
    FRM.setReceiver(receiver)
    FRM.setUpdater(updater)
    FRM.trigger()
    FRM.start()

    print("[INFO] Online inference with GUI started. Press Ctrl+C to quit.")
    try:
        sys.exit(app.exec_())
    except KeyboardInterrupt:
        pass
    finally:
        try:
            FRM.stop()
        except Exception:
            pass
        try:
            kgl.ksoclib.closeDevice()
        except Exception:
            pass
        print("[INFO] Stopped.")

if __name__ == "__main__":
    main()
