import configparser
import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    save_path: str = ""
    serial_port: str = ""
    baud_rate: str = "19200"
    serial_timeout: str = "1.0"
    pixels_per_mm: float = 100.0
    magnification: float = 1.0
    scale_curve_factor: float = 1.0


class ConfigManager:
    def __init__(self, settings_file, default_dir):
        self.settings_file = settings_file
        self.default_dir = default_dir
        self.config = AppConfig()

    def load(self):
        parser = configparser.ConfigParser()
        if os.path.exists(self.settings_file):
            parser.read(self.settings_file, encoding="utf-8")
            self.config.save_path = parser.get("Settings", "save_path", fallback="")
            self.config.serial_port = parser.get("Serial", "port", fallback="")
            self.config.baud_rate = parser.get("Serial", "baud_rate", fallback="19200")
            self.config.serial_timeout = parser.get("Serial", "timeout", fallback="1.0")
            try:
                self.config.pixels_per_mm = float(
                    parser.get("Scale", "pixels_per_mm", fallback="100.0")
                )
            except ValueError:
                self.config.pixels_per_mm = 100.0
            try:
                self.config.magnification = float(
                    parser.get("Scale", "magnification", fallback="1.0")
                )
            except ValueError:
                self.config.magnification = 1.0
            try:
                self.config.scale_curve_factor = float(
                    parser.get("Scale", "scale_curve_factor", fallback="1.0")
                )
            except ValueError:
                self.config.scale_curve_factor = 1.0
        if self.config.pixels_per_mm <= 0:
            self.config.pixels_per_mm = 100.0
        if self.config.magnification <= 0:
            self.config.magnification = 1.0
        if self.config.scale_curve_factor <= 0:
            self.config.scale_curve_factor = 1.0
        return self.config

    def save(self):
        parser = configparser.ConfigParser()
        parser["Settings"] = {"save_path": self.config.save_path}
        parser["Serial"] = {
            "port": self.config.serial_port,
            "baud_rate": self.config.baud_rate,
            "timeout": self.config.serial_timeout,
        }
        parser["Scale"] = {
            "pixels_per_mm": str(self.config.pixels_per_mm),
            "magnification": str(self.config.magnification),
            "scale_curve_factor": str(self.config.scale_curve_factor),
        }
        with open(self.settings_file, "w", encoding="utf-8") as file:
            parser.write(file)

    @property
    def save_path(self):
        return self.config.save_path

    @save_path.setter
    def save_path(self, value):
        self.config.save_path = value or ""

    @property
    def serial_port(self):
        return self.config.serial_port

    @serial_port.setter
    def serial_port(self, value):
        self.config.serial_port = value or ""

    @property
    def baud_rate(self):
        return self.config.baud_rate

    @baud_rate.setter
    def baud_rate(self, value):
        self.config.baud_rate = str(value)

    @property
    def serial_timeout(self):
        return self.config.serial_timeout

    @serial_timeout.setter
    def serial_timeout(self, value):
        self.config.serial_timeout = str(value)

    @property
    def pixels_per_mm(self):
        return self.config.pixels_per_mm

    @pixels_per_mm.setter
    def pixels_per_mm(self, value):
        self.config.pixels_per_mm = float(value)

    @property
    def magnification(self):
        return self.config.magnification

    @magnification.setter
    def magnification(self, value):
        self.config.magnification = float(value)

    @property
    def scale_curve_factor(self):
        return self.config.scale_curve_factor

    @scale_curve_factor.setter
    def scale_curve_factor(self, value):
        self.config.scale_curve_factor = float(value)

    def effective_save_path(self):
        return self.save_path or self.default_dir
