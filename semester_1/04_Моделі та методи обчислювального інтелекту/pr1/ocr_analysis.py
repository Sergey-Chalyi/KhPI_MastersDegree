#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Програма для аналізу якості розпізнавання тексту з використанням Tesseract та EasyOCR
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import cv2
import pytesseract
import easyocr
from difflib import SequenceMatcher
import pandas as pd
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

class OCRAnalyzer:
    def __init__(self):
        """Ініціалізація аналізатора OCR"""
        self.english_text = "Hello World! This is a test of OCR recognition quality."
        self.ukrainian_text = "Привіт Світ! Це тест якості розпізнавання тексту OCR."
        
        # Створюємо папку для результатів
        os.makedirs('results', exist_ok=True)
        os.makedirs('images', exist_ok=True)
        
        # Ініціалізуємо EasyOCR
        self.easyocr_reader = easyocr.Reader(['en', 'uk'])
        
    def create_background_image(self, width: int = 800, height: int = 600) -> Image.Image:
        """Створює зображення з фоном з різними фігурами"""
        # Створюємо біле зображення
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Кольори для фігур
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
        
        # Малюємо різні фігури
        # 1. Коло
        draw.ellipse([50, 50, 200, 200], fill=colors[0], outline='blue', width=3)
        
        # 2. Прямокутник
        draw.rectangle([300, 100, 500, 250], fill=colors[1], outline='green', width=3)
        
        # 3. Лінія
        draw.line([(100, 300), (400, 350)], fill='red', width=5)
        
        # 4. Трикутник (через полігон)
        triangle_points = [(600, 100), (700, 200), (500, 200)]
        draw.polygon(triangle_points, fill=colors[2], outline='purple', width=3)
        
        # 5. Еліпс
        draw.ellipse([150, 400, 350, 500], fill=colors[3], outline='orange', width=3)
        
        return img
    
    def add_text_to_image(self, background_img: Image.Image, text: str, 
                         font_size: int = 24, text_color: str = 'black') -> Image.Image:
        """Додає текст до зображення"""
        img = background_img.copy()
        draw = ImageDraw.Draw(img)
        
        try:
            # Спробуємо використати системний шрифт
            font = ImageFont.truetype("arial.ttf", font_size)
        except:
            try:
                # Альтернативний шрифт для Windows
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            except:
                # Використовуємо стандартний шрифт
                font = ImageFont.load_default()
        
        # Позиціонуємо текст по центру
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (img.width - text_width) // 2
        y = (img.height - text_height) // 2
        
        # Додаємо тінь для кращої видимості
        draw.text((x+2, y+2), text, font=font, fill='gray')
        draw.text((x, y), text, font=font, fill=text_color)
        
        return img
    
    def create_test_images(self):
        """Створює тестові зображення з текстом"""
        print("Створення тестових зображень...")
        
        # Створюємо фон
        background = self.create_background_image()
        background.save('images/background.png')
        
        # Створюємо зображення з англійським текстом
        english_img = self.add_text_to_image(background, self.english_text)
        english_img.save('images/english_text.png')
        
        # Створюємо зображення з українським текстом
        ukrainian_img = self.add_text_to_image(background, self.ukrainian_text)
        ukrainian_img.save('images/ukrainian_text.png')
        
        print("Тестові зображення створено!")
    
    def preprocess_image(self, image_path: str, threshold: int = 128) -> Dict[str, np.ndarray]:
        """Попередня обробка зображення для OCR"""
        # Завантажуємо зображення
        img = cv2.imread(image_path)
        
        # Кольорове зображення
        color_img = img.copy()
        
        # Зображення у відтінках сірого
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Бінаризоване зображення
        _, binary_img = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
        
        return {
            'color': color_img,
            'gray': gray_img,
            'binary': binary_img
        }
    
    def calculate_cer(self, reference: str, hypothesis: str) -> float:
        """Обчислює Character Error Rate (CER)"""
        if not reference:
            return 1.0 if hypothesis else 0.0
        
        matcher = SequenceMatcher(None, reference.lower(), hypothesis.lower())
        errors = len(reference) - matcher.matching_blocks[0].size if matcher.matching_blocks else len(reference)
        return errors / len(reference)
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Обчислює Word Error Rate (WER)"""
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if not ref_words:
            return 1.0 if hyp_words else 0.0
        
        matcher = SequenceMatcher(None, ref_words, hyp_words)
        errors = len(ref_words) - matcher.matching_blocks[0].size if matcher.matching_blocks else len(ref_words)
        return errors / len(ref_words)
    
    def tesseract_ocr(self, image: np.ndarray, language: str = 'eng') -> Tuple[str, float]:
        """Виконує розпізнавання тексту за допомогою Tesseract"""
        start_time = time.time()
        
        # Налаштування Tesseract
        config = '--oem 3 --psm 6'
        if language == 'ukr':
            config += ' -l ukr+eng'
        else:
            config += ' -l eng'
        
        # Розпізнавання тексту
        text = pytesseract.image_to_string(image, config=config)
        processing_time = time.time() - start_time
        
        return text.strip(), processing_time
    
    def easyocr_ocr(self, image: np.ndarray, language: str = 'en') -> Tuple[str, float]:
        """Виконує розпізнавання тексту за допомогою EasyOCR"""
        start_time = time.time()
        
        # Розпізнавання тексту
        results = self.easyocr_reader.readtext(image)
        text = ' '.join([result[1] for result in results])
        processing_time = time.time() - start_time
        
        return text.strip(), processing_time
    
    def get_word_coordinates(self, image: np.ndarray, language: str = 'eng') -> List[Dict]:
        """Отримує координати слів та їх обмежувальні рамки"""
        # Налаштування Tesseract для отримання координат
        config = '--oem 3 --psm 6'
        if language == 'ukr':
            config += ' -l ukr+eng'
        else:
            config += ' -l eng'
        
        # Отримуємо дані про слова
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        
        words_info = []
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 30:  # Фільтруємо слова з низькою впевненістю
                word_info = {
                    'text': data['text'][i],
                    'x': data['left'][i],
                    'y': data['top'][i],
                    'width': data['width'][i],
                    'height': data['height'][i],
                    'confidence': data['conf'][i]
                }
                words_info.append(word_info)
        
        return words_info
    
    def draw_bounding_boxes(self, image: np.ndarray, words_info: List[Dict], 
                           output_path: str):
        """Малює обмежувальні рамки навколо слів"""
        img_with_boxes = image.copy()
        
        for word in words_info:
            x, y, w, h = word['x'], word['y'], word['width'], word['height']
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img_with_boxes, word['text'], (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imwrite(output_path, img_with_boxes)
    
    def analyze_threshold_impact(self, image_path: str, reference_text: str, 
                                language: str = 'eng') -> pd.DataFrame:
        """Аналізує вплив порогу бінаризації на якість розпізнавання"""
        print(f"Аналіз впливу порогу бінаризації для {language}...")
        
        results = []
        thresholds = range(50, 251, 25)  # Пороги від 50 до 250 з кроком 25
        
        for threshold in thresholds:
            # Попередня обробка зображення
            processed_images = self.preprocess_image(image_path, threshold)
            
            # Розпізнавання для кожного типу зображення
            for img_type, img in processed_images.items():
                text, time_taken = self.tesseract_ocr(img, language)
                
                cer = self.calculate_cer(reference_text, text)
                wer = self.calculate_wer(reference_text, text)
                
                results.append({
                    'threshold': threshold,
                    'image_type': img_type,
                    'text': text,
                    'cer': cer,
                    'wer': wer,
                    'time': time_taken
                })
        
        return pd.DataFrame(results)
    
    def plot_threshold_analysis(self, df: pd.DataFrame, language: str):
        """Побудує графіки залежності метрик від порогу бінаризації"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Графік CER
        for img_type in df['image_type'].unique():
            data = df[df['image_type'] == img_type]
            ax1.plot(data['threshold'], data['cer'], marker='o', label=img_type)
        
        ax1.set_xlabel('Поріг бінаризації')
        ax1.set_ylabel('CER (Character Error Rate)')
        ax1.set_title(f'Залежність CER від порогу бінаризації ({language})')
        ax1.legend()
        ax1.grid(True)
        
        # Графік WER
        for img_type in df['image_type'].unique():
            data = df[df['image_type'] == img_type]
            ax2.plot(data['threshold'], data['wer'], marker='s', label=img_type)
        
        ax2.set_xlabel('Поріг бінаризації')
        ax2.set_ylabel('WER (Word Error Rate)')
        ax2.set_title(f'Залежність WER від порогу бінаризації ({language})')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'results/threshold_analysis_{language}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def find_optimal_threshold(self, df: pd.DataFrame) -> Dict:
        """Знаходить оптимальний поріг бінаризації"""
        # Знаходимо найкращий результат для бінаризованого зображення
        binary_data = df[df['image_type'] == 'binary']
        
        # Знаходимо поріг з найменшою сумою CER та WER
        binary_data['combined_score'] = binary_data['cer'] + binary_data['wer']
        optimal_idx = binary_data['combined_score'].idxmin()
        
        return {
            'threshold': binary_data.loc[optimal_idx, 'threshold'],
            'cer': binary_data.loc[optimal_idx, 'cer'],
            'wer': binary_data.loc[optimal_idx, 'wer'],
            'combined_score': binary_data.loc[optimal_idx, 'combined_score']
        }
    
    def compare_ocr_engines(self, image_path: str, reference_text: str, 
                           language: str = 'eng') -> Dict:
        """Порівнює Tesseract та EasyOCR"""
        print(f"Порівняння OCR двигунів для {language}...")
        
        # Попередня обробка зображення
        processed_images = self.preprocess_image(image_path, 128)
        
        results = {}
        
        for img_type, img in processed_images.items():
            # Tesseract
            tesseract_text, tesseract_time = self.tesseract_ocr(img, language)
            tesseract_cer = self.calculate_cer(reference_text, tesseract_text)
            tesseract_wer = self.calculate_wer(reference_text, tesseract_text)
            
            # EasyOCR
            easyocr_lang = 'uk' if language == 'ukr' else 'en'
            easyocr_text, easyocr_time = self.easyocr_ocr(img, easyocr_lang)
            easyocr_cer = self.calculate_cer(reference_text, easyocr_text)
            easyocr_wer = self.calculate_wer(reference_text, easyocr_text)
            
            results[img_type] = {
                'tesseract': {
                    'text': tesseract_text,
                    'cer': tesseract_cer,
                    'wer': tesseract_wer,
                    'time': tesseract_time
                },
                'easyocr': {
                    'text': easyocr_text,
                    'cer': easyocr_cer,
                    'wer': easyocr_wer,
                    'time': easyocr_time
                }
            }
        
        return results
    
    def create_comparison_table(self, english_results: Dict, ukrainian_results: Dict) -> str:
        """Створює порівняльну таблицю в форматі Markdown"""
        table = "# Порівняльна таблиця OCR двигунів\n\n"
        
        for img_type in ['color', 'gray', 'binary']:
            table += f"## {img_type.upper()} зображення\n\n"
            table += "| Метрика | Tesseract (EN) | EasyOCR (EN) | Tesseract (UK) | EasyOCR (UK) |\n"
            table += "|---------|----------------|--------------|----------------|--------------|\n"
            
            # CER
            table += f"| CER | {english_results[img_type]['tesseract']['cer']:.3f} | "
            table += f"{english_results[img_type]['easyocr']['cer']:.3f} | "
            table += f"{ukrainian_results[img_type]['tesseract']['cer']:.3f} | "
            table += f"{ukrainian_results[img_type]['easyocr']['cer']:.3f} |\n"
            
            # WER
            table += f"| WER | {english_results[img_type]['tesseract']['wer']:.3f} | "
            table += f"{english_results[img_type]['easyocr']['wer']:.3f} | "
            table += f"{ukrainian_results[img_type]['tesseract']['wer']:.3f} | "
            table += f"{ukrainian_results[img_type]['easyocr']['wer']:.3f} |\n"
            
            # Час
            table += f"| Час (с) | {english_results[img_type]['tesseract']['time']:.3f} | "
            table += f"{english_results[img_type]['easyocr']['time']:.3f} | "
            table += f"{ukrainian_results[img_type]['tesseract']['time']:.3f} | "
            table += f"{ukrainian_results[img_type]['easyocr']['time']:.3f} |\n\n"
        
        return table
    
    def run_full_analysis(self):
        """Запускає повний аналіз"""
        print("Початок повного аналізу OCR...")
        
        # 1. Створюємо тестові зображення
        self.create_test_images()
        
        # 2. Аналіз для англійської мови
        print("\n=== Аналіз для англійської мови ===")
        english_df = self.analyze_threshold_impact('images/english_text.png', 
                                                  self.english_text, 'eng')
        self.plot_threshold_analysis(english_df, 'english')
        english_optimal = self.find_optimal_threshold(english_df)
        print(f"Оптимальний поріг для англійської мови: {english_optimal}")
        
        # 3. Аналіз для української мови
        print("\n=== Аналіз для української мови ===")
        ukrainian_df = self.analyze_threshold_impact('images/ukrainian_text.png', 
                                                    self.ukrainian_text, 'ukr')
        self.plot_threshold_analysis(ukrainian_df, 'ukrainian')
        ukrainian_optimal = self.find_optimal_threshold(ukrainian_df)
        print(f"Оптимальний поріг для української мови: {ukrainian_optimal}")
        
        # 4. Порівняння OCR двигунів
        print("\n=== Порівняння OCR двигунів ===")
        english_comparison = self.compare_ocr_engines('images/english_text.png', 
                                                     self.english_text, 'eng')
        ukrainian_comparison = self.compare_ocr_engines('images/ukrainian_text.png', 
                                                       self.ukrainian_text, 'ukr')
        
        # 5. Створюємо порівняльну таблицю
        comparison_table = self.create_comparison_table(english_comparison, ukrainian_comparison)
        with open('results/comparison_table.md', 'w', encoding='utf-8') as f:
            f.write(comparison_table)
        
        # 6. Отримуємо координати слів та малюємо рамки
        print("\n=== Отримання координат слів ===")
        for lang, lang_code in [('english', 'eng'), ('ukrainian', 'ukr')]:
            img_path = f'images/{lang}_text.png'
            processed_images = self.preprocess_image(img_path, 128)
            
            for img_type, img in processed_images.items():
                words_info = self.get_word_coordinates(img, lang_code)
                output_path = f'results/{lang}_{img_type}_bounding_boxes.png'
                self.draw_bounding_boxes(img, words_info, output_path)
                print(f"Збережено: {output_path}")
        
        print("\nАналіз завершено! Результати збережено в папці 'results/'")

def main():
    """Головна функція"""
    analyzer = OCRAnalyzer()
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
