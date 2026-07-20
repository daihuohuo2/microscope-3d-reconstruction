from PyQt5.QtCore import QObject, QEvent, QPoint, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget


class ScaleBarOverlay(QWidget):
    BAR_WIDTH_FRACTION = 0.20

    def __init__(self, target_widget):
        # Must be a top-level window so that WA_TranslucentBackground works via
        # DWM compositing on Windows. A child/sibling widget's transparent areas
        # only show the parent background, not the SDK-rendered GDI content.
        super().__init__(
            None,
            Qt.Tool | Qt.FramelessWindowHint | Qt.WindowTransparentForInput
            | Qt.WindowDoesNotAcceptFocus,
        )
        self._target_widget = target_widget
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self._pixels_per_mm = 100.0
        self._img_width = 0
        self._bar_visible = False
        self.hide()

    def set_pixels_per_mm(self, value):
        self._pixels_per_mm = max(float(value), 0.001)
        self.update()

    def set_img_width(self, width):
        self._img_width = int(width)
        self.update()

    def set_visible(self, visible):
        self._bar_visible = bool(visible)
        if self._bar_visible:
            self.update_size()
            self.show()
            self.raise_()
        else:
            self.hide()

    def update_size(self):
        tw = self._target_widget
        if not tw.isVisible():
            return
        tl = tw.mapToGlobal(QPoint(0, 0))
        self.setGeometry(tl.x(), tl.y(), tw.width(), tw.height())
        self.update()

    def paintEvent(self, event):
        if not self._bar_visible or self._pixels_per_mm <= 0:
            return
        display_w = self.width()
        display_h = self.height()
        if display_w <= 0 or display_h <= 0:
            return

        scale_factor = display_w / self._img_width if self._img_width > 0 else 1.0
        display_ppmm = self._pixels_per_mm * scale_factor
        bar_len_px = max(display_w * self.BAR_WIDTH_FRACTION, 4.0)
        bar_len_mm = bar_len_px / max(display_ppmm, 0.001)

        bar_len_um = bar_len_mm * 1000.0
        if bar_len_um < 10.0:
            label = "{:.2f} um".format(bar_len_um)
        elif bar_len_um < 100.0:
            label = "{:.1f} um".format(bar_len_um)
        else:
            label = "{:.0f} um".format(bar_len_um)

        margin = 16
        bar_h = 8
        x0 = max(margin, display_w - margin - int(bar_len_px))
        y0 = display_h - margin - bar_h - 18
        x1 = x0 + int(bar_len_px)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.drawRect(x0 - 2, y0 - 2, int(bar_len_px) + 4, bar_h + 4)
        painter.setBrush(QColor(255, 255, 255, 230))
        painter.drawRect(x0, y0, int(bar_len_px), bar_h)
        painter.setPen(QPen(QColor(255, 255, 255, 230), 2))
        tick_h = bar_h + 4
        painter.drawLine(x0, y0 - 2, x0, y0 + tick_h)
        painter.drawLine(x1, y0 - 2, x1, y0 + tick_h)
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.setPen(QColor(0, 0, 0, 200))
        painter.drawText(x0 + 1, y0 - 3, label)
        painter.setPen(QColor(255, 255, 255, 240))
        painter.drawText(x0, y0 - 4, label)
        painter.end()


class ResizeFilter(QObject):
    def __init__(self, overlay):
        super().__init__()
        self._overlay = overlay

    def eventFilter(self, obj, event):
        if event.type() in (QEvent.Resize, QEvent.Move):
            self._overlay.update_size()
        return False


class DoubleClickFilter(QObject):
    """Event filter that calls a callback(wx, wy) on mouse double-click."""

    def __init__(self, callback):
        super().__init__()
        self._callback = callback

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonDblClick:
            self._callback(event.x(), event.y())
        return False
