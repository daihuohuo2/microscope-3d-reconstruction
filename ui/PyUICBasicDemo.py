# -*- coding: utf-8 -*-

# Layout: Camera (left, stretch) | Right-Panel (two columns, fixed width)
#   Right-Col-1: 初始化 / 采集(含自动对焦) / 参数 / 自动拍摄
#   Right-Col-2: 串口设置 / 运动控制 / 比例尺 / 底噪扣除
# All groups visible at default window size (~1200x700) without scrolling.

from PyQt5 import QtCore, QtGui, QtWidgets

_COMPACT_MARGINS  = (8, 14, 8, 8)
_COMPACT_SPACING  = 4
_COL_MAX_W        = 280
_COL_MIN_W        = 240


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1200, 700)
        MainWindow.setMinimumSize(960, 580)

        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")

        mainH = QtWidgets.QHBoxLayout(self.centralWidget)
        mainH.setContentsMargins(8, 8, 8, 8)
        mainH.setSpacing(8)

        #  左侧：设备下拉 + 相机预览 
        leftV = QtWidgets.QVBoxLayout()
        leftV.setSpacing(6)

        self.ComboDevices = QtWidgets.QComboBox(self.centralWidget)
        self.ComboDevices.setObjectName("ComboDevices")
        leftV.addWidget(self.ComboDevices)

        self.widgetDisplay = QtWidgets.QWidget(self.centralWidget)
        self.widgetDisplay.setObjectName("widgetDisplay")
        self.widgetDisplay.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding)
        self.widgetDisplay.setMinimumSize(320, 200)
        self.widgetDisplay.setStyleSheet("background: #111;")
        leftV.addWidget(self.widgetDisplay, stretch=1)

        mainH.addLayout(leftV, stretch=1)

        #  右侧双列 
        rightH = QtWidgets.QHBoxLayout()
        rightH.setSpacing(6)
        rightH.setContentsMargins(0, 0, 0, 0)

        col1V = QtWidgets.QVBoxLayout()
        col1V.setSpacing(5)
        col1V.addWidget(self._make_init_group())
        col1V.addWidget(self._make_grab_group())
        col1V.addWidget(self._make_param_group())
        col1V.addWidget(self._make_capture_group())
        col1V.addStretch(1)
        col1W = QtWidgets.QWidget()
        col1W.setLayout(col1V)
        col1W.setMinimumWidth(_COL_MIN_W)
        col1W.setMaximumWidth(_COL_MAX_W)
        rightH.addWidget(col1W)

        col2V = QtWidgets.QVBoxLayout()
        col2V.setSpacing(5)
        col2V.addWidget(self._make_serial_group())
        col2V.addWidget(self._make_motion_group())
        col2V.addWidget(self._make_scale_group())
        col2V.addWidget(self._make_dark_group())
        col2V.addStretch(1)
        col2W = QtWidgets.QWidget()
        col2W.setLayout(col2V)
        col2W.setMinimumWidth(_COL_MIN_W)
        col2W.setMaximumWidth(_COL_MAX_W)
        rightH.addWidget(col2W)

        rightContainer = QtWidgets.QWidget()
        rightContainer.setLayout(rightH)
        mainH.addWidget(rightContainer, stretch=0)

        MainWindow.setCentralWidget(self.centralWidget)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    #  group helpers 

    def _make_init_group(self):
        grp = QtWidgets.QGroupBox()
        grp.setObjectName("groupInit")
        g = QtWidgets.QGridLayout(grp)
        g.setContentsMargins(*_COMPACT_MARGINS)
        g.setSpacing(_COMPACT_SPACING)

        self.bnEnum = QtWidgets.QPushButton()
        self.bnEnum.setObjectName("bnEnum")
        g.addWidget(self.bnEnum, 0, 0, 1, 2)

        self.bnOpen = QtWidgets.QPushButton()
        self.bnOpen.setObjectName("bnOpen")
        g.addWidget(self.bnOpen, 1, 0)

        self.bnClose = QtWidgets.QPushButton()
        self.bnClose.setEnabled(False)
        self.bnClose.setObjectName("bnClose")
        g.addWidget(self.bnClose, 1, 1)

        g.setColumnStretch(0, 1); g.setColumnStretch(1, 1)
        self.groupInit = grp
        return grp

    def _make_grab_group(self):
        grp = QtWidgets.QGroupBox()
        grp.setEnabled(False)
        grp.setObjectName("groupGrab")
        g = QtWidgets.QGridLayout(grp)
        g.setContentsMargins(*_COMPACT_MARGINS)
        g.setSpacing(_COMPACT_SPACING)

        self.bnStart = QtWidgets.QPushButton()
        self.bnStart.setEnabled(False)
        self.bnStart.setObjectName("bnStart")
        g.addWidget(self.bnStart, 0, 0)

        self.bnStop = QtWidgets.QPushButton()
        self.bnStop.setEnabled(False)
        self.bnStop.setObjectName("bnStop")
        g.addWidget(self.bnStop, 0, 1)

        self.bnAutoFocus = QtWidgets.QPushButton()
        self.bnAutoFocus.setEnabled(False)
        self.bnAutoFocus.setObjectName("bnAutoFocus")
        g.addWidget(self.bnAutoFocus, 1, 0)

        self.bnStopAutoFocus = QtWidgets.QPushButton()
        self.bnStopAutoFocus.setEnabled(False)
        self.bnStopAutoFocus.setObjectName("bnStopAutoFocus")
        g.addWidget(self.bnStopAutoFocus, 1, 1)

        self.lblAutoFocusStatus = QtWidgets.QLabel()
        self.lblAutoFocusStatus.setObjectName("lblAutoFocusStatus")
        self.lblAutoFocusStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.lblAutoFocusStatus.setStyleSheet("font-size: 11px; color: #555;")
        g.addWidget(self.lblAutoFocusStatus, 2, 0, 1, 2)

        g.setColumnStretch(0, 1); g.setColumnStretch(1, 1)
        self.groupGrab = grp
        return grp

    def _make_param_group(self):
        grp = QtWidgets.QGroupBox()
        grp.setEnabled(False)
        grp.setObjectName("groupParam")
        g = QtWidgets.QGridLayout(grp)
        g.setContentsMargins(*_COMPACT_MARGINS)
        g.setSpacing(_COMPACT_SPACING)

        self.label_4 = QtWidgets.QLabel(); self.label_4.setObjectName("label_4")
        self.edtExposureTime = QtWidgets.QLineEdit(); self.edtExposureTime.setObjectName("edtExposureTime")
        g.addWidget(self.label_4, 0, 0); g.addWidget(self.edtExposureTime, 0, 1)

        self.label_5 = QtWidgets.QLabel(); self.label_5.setObjectName("label_5")
        self.edtGain = QtWidgets.QLineEdit(); self.edtGain.setObjectName("edtGain")
        g.addWidget(self.label_5, 1, 0); g.addWidget(self.edtGain, 1, 1)

        self.label_6 = QtWidgets.QLabel(); self.label_6.setObjectName("label_6")
        self.edtFrameRate = QtWidgets.QLineEdit(); self.edtFrameRate.setObjectName("edtFrameRate")
        g.addWidget(self.label_6, 2, 0); g.addWidget(self.edtFrameRate, 2, 1)

        self.bnGetParam = QtWidgets.QPushButton(); self.bnGetParam.setObjectName("bnGetParam")
        self.bnSetParam = QtWidgets.QPushButton(); self.bnSetParam.setObjectName("bnSetParam")
        g.addWidget(self.bnGetParam, 3, 0); g.addWidget(self.bnSetParam, 3, 1)

        g.setColumnStretch(0, 2); g.setColumnStretch(1, 3)
        self.groupParam = grp
        return grp

    def _make_capture_group(self):
        grp = QtWidgets.QGroupBox()
        grp.setObjectName("groupAutoCapture")
        g = QtWidgets.QGridLayout(grp)
        g.setContentsMargins(*_COMPACT_MARGINS)
        g.setSpacing(_COMPACT_SPACING)

        self.label_capture = QtWidgets.QLabel(); self.label_capture.setObjectName("label_capture")
        self.edtCaptureCount = QtWidgets.QLineEdit(); self.edtCaptureCount.setObjectName("edtCaptureCount")
        g.addWidget(self.label_capture, 0, 0); g.addWidget(self.edtCaptureCount, 0, 1)

        self.bnAutoCapture = QtWidgets.QPushButton()
        self.bnAutoCapture.setEnabled(False)
        self.bnAutoCapture.setObjectName("bnAutoCapture")
        g.addWidget(self.bnAutoCapture, 1, 0, 1, 2)

        self.bnSetSavePath = QtWidgets.QPushButton()
        self.bnSetSavePath.setObjectName("bnSetSavePath")
        g.addWidget(self.bnSetSavePath, 2, 0, 1, 2)

        self.lblSavePathInfo = QtWidgets.QLineEdit()
        self.lblSavePathInfo.setObjectName("lblSavePathInfo")
        self.lblSavePathInfo.setReadOnly(True)
        self.lblSavePathInfo.setStyleSheet("font-size: 10px; color: #555;")
        g.addWidget(self.lblSavePathInfo, 3, 0, 1, 2)

        g.setColumnStretch(0, 2); g.setColumnStretch(1, 3)
        self.groupAutoCapture = grp
        return grp

    def _make_serial_group(self):
        grp = QtWidgets.QGroupBox()
        grp.setObjectName("groupSerial")
        g = QtWidgets.QGridLayout(grp)
        g.setContentsMargins(*_COMPACT_MARGINS)
        g.setSpacing(_COMPACT_SPACING)

        self.label_serial_port = QtWidgets.QLabel(); self.label_serial_port.setObjectName("label_serial_port")
        self.cmbSerialPort = QtWidgets.QComboBox();   self.cmbSerialPort.setObjectName("cmbSerialPort")
        self.bnRefreshPort = QtWidgets.QPushButton(); self.bnRefreshPort.setObjectName("bnRefreshPort")
        g.addWidget(self.label_serial_port, 0, 0)
        g.addWidget(self.cmbSerialPort,     0, 1)
        g.addWidget(self.bnRefreshPort,     0, 2)

        self.label_baud_rate = QtWidgets.QLabel(); self.label_baud_rate.setObjectName("label_baud_rate")
        self.cmbBaudRate = QtWidgets.QComboBox();  self.cmbBaudRate.setObjectName("cmbBaudRate")
        g.addWidget(self.label_baud_rate, 1, 0); g.addWidget(self.cmbBaudRate, 1, 1, 1, 2)

        self.label_timeout = QtWidgets.QLabel(); self.label_timeout.setObjectName("label_timeout")
        self.edtSerialTimeout = QtWidgets.QLineEdit(); self.edtSerialTimeout.setObjectName("edtSerialTimeout")
        g.addWidget(self.label_timeout, 2, 0); g.addWidget(self.edtSerialTimeout, 2, 1, 1, 2)

        self.bnConnectSerial = QtWidgets.QPushButton(); self.bnConnectSerial.setObjectName("bnConnectSerial")
        self.lblSerialStatus = QtWidgets.QLabel();      self.lblSerialStatus.setObjectName("lblSerialStatus")
        self.lblSerialStatus.setAlignment(QtCore.Qt.AlignCenter)
        g.addWidget(self.bnConnectSerial, 3, 0, 1, 2); g.addWidget(self.lblSerialStatus, 3, 2)

        g.setColumnStretch(0, 2); g.setColumnStretch(1, 3); g.setColumnStretch(2, 2)
        self.groupSerial = grp
        return grp

    def _make_motion_group(self):
        grp = QtWidgets.QGroupBox()
        grp.setObjectName("groupMotion")
        g = QtWidgets.QGridLayout(grp)
        g.setContentsMargins(*_COMPACT_MARGINS)
        g.setSpacing(_COMPACT_SPACING)

        self.bnHomeZ = QtWidgets.QPushButton(); self.bnHomeZ.setObjectName("bnHomeZ")
        g.addWidget(self.bnHomeZ, 0, 0, 1, 3)

        self.bnCoarseUp = QtWidgets.QPushButton(); self.bnCoarseUp.setObjectName("bnCoarseUp")
        self.bnCoarseDown = QtWidgets.QPushButton(); self.bnCoarseDown.setObjectName("bnCoarseDown")
        g.addWidget(self.bnCoarseUp, 1, 0, 1, 2)
        g.addWidget(self.bnCoarseDown, 1, 2, 1, 1)

        self.bnMediumUp = QtWidgets.QPushButton(); self.bnMediumUp.setObjectName("bnMediumUp")
        self.bnMediumDown = QtWidgets.QPushButton(); self.bnMediumDown.setObjectName("bnMediumDown")
        g.addWidget(self.bnMediumUp, 2, 0, 1, 2)
        g.addWidget(self.bnMediumDown, 2, 2, 1, 1)

        self.bnFineUp = QtWidgets.QPushButton(); self.bnFineUp.setObjectName("bnFineUp")
        self.bnFineDown = QtWidgets.QPushButton(); self.bnFineDown.setObjectName("bnFineDown")
        g.addWidget(self.bnFineUp, 3, 0, 1, 2)
        g.addWidget(self.bnFineDown, 3, 2, 1, 1)

        self.bnMoveStep = QtWidgets.QPushButton(); self.bnMoveStep.setObjectName("bnMoveStep")
        self.bnMoveStepDown = QtWidgets.QPushButton(); self.bnMoveStepDown.setObjectName("bnMoveStepDown")
        g.addWidget(self.bnMoveStep, 4, 0, 1, 2)
        g.addWidget(self.bnMoveStepDown, 4, 2, 1, 1)

        self.label_light = QtWidgets.QLabel(); self.label_light.setObjectName("label_light")
        self.sliderLight = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sliderLight.setObjectName("sliderLight")
        self.sliderLight.setRange(0, 255)
        self.sliderLight.setValue(255)
        self.lblLightValue = QtWidgets.QLabel("255")
        self.lblLightValue.setObjectName("lblLightValue")
        self.lblLightValue.setMinimumWidth(28)
        self.lblLightValue.setAlignment(QtCore.Qt.AlignCenter)
        g.addWidget(self.label_light, 5, 0); g.addWidget(self.sliderLight, 5, 1); g.addWidget(self.lblLightValue, 5, 2)

        g.setColumnStretch(0, 2); g.setColumnStretch(1, 3); g.setColumnStretch(2, 2)
        self.groupMotion = grp
        return grp

    def _make_scale_group(self):
        grp = QtWidgets.QGroupBox()
        grp.setObjectName("groupScaleBar")
        g = QtWidgets.QGridLayout(grp)
        g.setContentsMargins(*_COMPACT_MARGINS)
        g.setSpacing(_COMPACT_SPACING)

        self.chkShowScaleBar = QtWidgets.QCheckBox()
        self.chkShowScaleBar.setObjectName("chkShowScaleBar")
        g.addWidget(self.chkShowScaleBar, 0, 0)

        self.label_ppmm = QtWidgets.QLabel(); self.label_ppmm.setObjectName("label_ppmm")
        self.edtPixelsPerMm = QtWidgets.QLineEdit(); self.edtPixelsPerMm.setObjectName("edtPixelsPerMm")
        g.addWidget(self.label_ppmm,     0, 1)
        g.addWidget(self.edtPixelsPerMm, 0, 2)

        self.bnSetScaleCalib = QtWidgets.QPushButton()
        self.bnSetScaleCalib.setObjectName("bnSetScaleCalib")
        g.addWidget(self.bnSetScaleCalib, 1, 0)

        self.lblScaleBarInfo = QtWidgets.QLabel()
        self.lblScaleBarInfo.setObjectName("lblScaleBarInfo")
        self.lblScaleBarInfo.setAlignment(QtCore.Qt.AlignCenter)
        self.lblScaleBarInfo.setStyleSheet("font-size: 10px; color: #444;")
        g.addWidget(self.lblScaleBarInfo, 1, 1, 1, 2)

        self.lblAutoCalibSep = QtWidgets.QLabel()
        self.lblAutoCalibSep.setObjectName("lblAutoCalibSep")
        self.lblAutoCalibSep.setAlignment(QtCore.Qt.AlignCenter)
        self.lblAutoCalibSep.setStyleSheet("color: gray; font-size: 10px;")
        g.addWidget(self.lblAutoCalibSep, 2, 0, 1, 3)

        self.label_calib_move = QtWidgets.QLabel(); self.label_calib_move.setObjectName("label_calib_move")
        self.edtCalibMoveMm = QtWidgets.QLineEdit(); self.edtCalibMoveMm.setObjectName("edtCalibMoveMm")
        g.addWidget(self.label_calib_move, 3, 0); g.addWidget(self.edtCalibMoveMm, 3, 1, 1, 2)

        self.bnAutoCalib = QtWidgets.QPushButton()
        self.bnAutoCalib.setEnabled(False)
        self.bnAutoCalib.setObjectName("bnAutoCalib")
        g.addWidget(self.bnAutoCalib, 4, 0)

        self.lblAutoCalibStatus = QtWidgets.QLabel()
        self.lblAutoCalibStatus.setObjectName("lblAutoCalibStatus")
        self.lblAutoCalibStatus.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.lblAutoCalibStatus.setWordWrap(True)
        self.lblAutoCalibStatus.setStyleSheet("font-size: 10px; color: #444;")
        g.addWidget(self.lblAutoCalibStatus, 4, 1, 1, 2)

        g.setColumnStretch(0, 3); g.setColumnStretch(1, 2); g.setColumnStretch(2, 3)
        self.groupScaleBar = grp
        return grp

    def _make_dark_group(self):
        grp = QtWidgets.QGroupBox()
        grp.setObjectName("groupDarkSub")
        g = QtWidgets.QGridLayout(grp)
        g.setContentsMargins(*_COMPACT_MARGINS)
        g.setSpacing(_COMPACT_SPACING)

        self.bnCaptureDark = QtWidgets.QPushButton()
        self.bnCaptureDark.setEnabled(False)
        self.bnCaptureDark.setObjectName("bnCaptureDark")
        g.addWidget(self.bnCaptureDark, 0, 0, 1, 2)

        self.chkDarkSub = QtWidgets.QCheckBox()
        self.chkDarkSub.setEnabled(False)
        self.chkDarkSub.setObjectName("chkDarkSub")
        g.addWidget(self.chkDarkSub, 1, 0)

        self.bnClearDark = QtWidgets.QPushButton()
        self.bnClearDark.setEnabled(False)
        self.bnClearDark.setObjectName("bnClearDark")
        g.addWidget(self.bnClearDark, 1, 1)

        self.lblDarkSubStatus = QtWidgets.QLabel()
        self.lblDarkSubStatus.setObjectName("lblDarkSubStatus")
        self.lblDarkSubStatus.setAlignment(QtCore.Qt.AlignCenter)
        self.lblDarkSubStatus.setWordWrap(True)
        self.lblDarkSubStatus.setStyleSheet("font-size: 10px; color: #555;")
        g.addWidget(self.lblDarkSubStatus, 2, 0, 1, 2)

        g.setColumnStretch(0, 1); g.setColumnStretch(1, 1)
        self.groupDarkSub = grp
        return grp

    #  文本设置 
    def retranslateUi(self, MainWindow):
        _ = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_("MainWindow", "MainWindow"))
        self.groupInit.setTitle(            _("MainWindow", "初始化"))
        self.bnEnum.setText(                _("MainWindow", "查找设备"))
        self.bnOpen.setText(                _("MainWindow", "打开设备"))
        self.bnClose.setText(               _("MainWindow", "关闭设备"))
        self.groupGrab.setTitle(            _("MainWindow", "采集"))
        self.bnStart.setText(               _("MainWindow", "开始采集"))
        self.bnStop.setText(                _("MainWindow", "停止采集"))
        self.bnAutoFocus.setText(           _("MainWindow", "开始自动对焦"))
        self.bnStopAutoFocus.setText(       _("MainWindow", "停止对焦"))
        self.lblAutoFocusStatus.setText(    _("MainWindow", "就绪"))
        self.groupParam.setTitle(           _("MainWindow", "参数"))
        self.label_4.setText(               _("MainWindow", "曝光"))
        self.edtExposureTime.setText(       _("MainWindow", "0"))
        self.label_5.setText(               _("MainWindow", "增益"))
        self.edtGain.setText(               _("MainWindow", "0"))
        self.label_6.setText(               _("MainWindow", "帧率"))
        self.edtFrameRate.setText(          _("MainWindow", "0"))
        self.bnGetParam.setText(            _("MainWindow", "获取参数"))
        self.bnSetParam.setText(            _("MainWindow", "设置参数"))
        self.groupAutoCapture.setTitle(     _("MainWindow", "自动拍摄"))
        self.label_capture.setText(         _("MainWindow", "张数"))
        self.edtCaptureCount.setText(       _("MainWindow", "1"))
        self.bnAutoCapture.setText(         _("MainWindow", "开始自动拍摄"))
        self.bnSetSavePath.setText(         _("MainWindow", "设置保存路径"))
        self.lblSavePathInfo.setText(       _("MainWindow", "保存至: (默认)"))
        self.groupSerial.setTitle(          _("MainWindow", "串口设置"))
        self.label_serial_port.setText(     _("MainWindow", "串口名称"))
        self.bnRefreshPort.setText(         _("MainWindow", "刷新串口"))
        self.label_baud_rate.setText(       _("MainWindow", "波特率"))
        self.label_timeout.setText(         _("MainWindow", "超时 (s)"))
        self.edtSerialTimeout.setText(      _("MainWindow", "1.0"))
        self.bnConnectSerial.setText(       _("MainWindow", "连接串口"))
        self.lblSerialStatus.setText(       _("MainWindow", " 未连接"))
        self.groupMotion.setTitle(          _("MainWindow", "运动控制"))
        self.bnHomeZ.setText(               _("MainWindow", "Z 轴归零"))
        self.bnCoarseUp.setText(            _("MainWindow", "Z 粗调（+1.00mm）"))
        self.bnCoarseDown.setText(          _("MainWindow", "粗调（-1.00mm）"))
        self.bnMediumUp.setText(            _("MainWindow", "Z 中调（+0.10mm）"))
        self.bnMediumDown.setText(          _("MainWindow", "中调（-0.10mm）"))
        self.bnFineUp.setText(              _("MainWindow", "Z 细调（+0.05mm）"))
        self.bnFineDown.setText(            _("MainWindow", "细调（-0.05mm）"))
        self.bnMoveStep.setText(            _("MainWindow", "Z 极细调（+0.005mm）"))
        self.bnMoveStepDown.setText(        _("MainWindow", "极细调（-0.005mm）"))
        self.label_light.setText(           _("MainWindow", "亮度"))
        self.groupScaleBar.setTitle(        _("MainWindow", "比例尺"))
        self.chkShowScaleBar.setText(       _("MainWindow", "显示比例尺"))
        self.label_ppmm.setText(            _("MainWindow", "像素/mm"))
        self.edtPixelsPerMm.setText(        _("MainWindow", "100.0"))
        self.bnSetScaleCalib.setText(       _("MainWindow", "应用标定值"))
        self.lblScaleBarInfo.setText(       _("MainWindow", ""))
        self.lblAutoCalibSep.setText(       _("MainWindow", " 自动标定 "))
        self.label_calib_move.setText(      _("MainWindow", "移动距离(mm)"))
        self.edtCalibMoveMm.setText(        _("MainWindow", "1.0"))
        self.bnAutoCalib.setText(           _("MainWindow", "自动标定"))
        self.lblAutoCalibStatus.setText(    _("MainWindow", "就绪"))
        self.groupDarkSub.setTitle(         _("MainWindow", "底噪扣除"))
        self.bnCaptureDark.setText(         _("MainWindow", "采集底噪帧"))
        self.chkDarkSub.setText(            _("MainWindow", "启用底噪扣除"))
        self.bnClearDark.setText(           _("MainWindow", "清除底噪帧"))
        self.lblDarkSubStatus.setText(      _("MainWindow", "未采集"))