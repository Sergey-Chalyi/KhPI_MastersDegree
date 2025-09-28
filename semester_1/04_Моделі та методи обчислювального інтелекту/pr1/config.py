#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Файл конфігурації для OCR аналізатора
"""

import os
import pytesseract

# Налаштування шляхів до Tesseract
# Розкоментуйте та змініть шлях відповідно до вашої системи

# Windows
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'D:\Programs\tesseract\tesseract.exe'
    pass

# Linux
elif os.name == 'posix':
    # pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    pass

# macOS
elif os.name == 'darwin':
    # pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'
    pass

# Налаштування тестових текстів
TEST_TEXTS = {
    'english': "Hello World! This is a test of OCR recognition quality.",
    'ukrainian': "Привіт Світ! Це тест якості розпізнавання тексту OCR."
}

# Налаштування зображень
IMAGE_CONFIG = {
    'width': 800,
    'height': 600,
    'font_size': 24,
    'text_color': 'black'
}

# Налаштування аналізу
ANALYSIS_CONFIG = {
    'threshold_range': (50, 251, 25),  # (start, stop, step)
    'confidence_threshold': 30,  # Мінімальна впевненість для слів
    'languages': {
        'english': 'eng',
        'ukrainian': 'ukr'
    }
}

# Кольори для фігур фону
BACKGROUND_COLORS = [
    'lightblue',
    'lightgreen', 
    'lightcoral',
    'lightyellow',
    'lightpink'
]

# Налаштування графіків
PLOT_CONFIG = {
    'figsize': (15, 6),
    'dpi': 300,
    'bbox_inches': 'tight'
}
