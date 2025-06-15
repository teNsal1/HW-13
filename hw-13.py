import os
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import requests
import base64

class RequestStrategy(ABC):
    """Абстрактный класс стратегии для запросов к Mistral API"""
    @abstractmethod
    def execute(self, text: str, model: str, history: Optional[List] = None, 
                image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Выполняет запрос к API
        
        Args:
            text: Текст запроса
            model: Используемая модель
            history: История сообщений (опционально)
            image_path: Путь к изображению (опционально)
            
        Returns:
            Ответ API в виде словаря
        """
        pass

class TextRequestStrategy(RequestStrategy):
    """Конкретная стратегия для текстовых запросов"""
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"

    def execute(self, text: str, model: str, history: Optional[List] = None, 
                image_path: Optional[str] = None) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Формируем сообщения с учетом истории
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": text})
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Text request error: {e}")
            return {"error": str(e)}

class ImageRequestStrategy(RequestStrategy):
    """Конкретная стратегия для мультимодальных запросов (текст + изображение)"""
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.base_url = "https://api.mistral.ai/v1/chat/completions"

    def _encode_image(self, image_path: str) -> Optional[str]:
        """Кодирует изображение в base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logging.error(f"Image encoding error: {e}")
            return None

    def execute(self, text: str, model: str, history: Optional[List] = None, 
                image_path: Optional[str] = None) -> Dict[str, Any]:
        if not image_path:
            return {"error": "Image path is required for image strategy"}
        
        base64_image = self._encode_image(image_path)
        if not base64_image:
            return {"error": "Failed to encode image"}
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Формируем мультимодальное сообщение
        content = [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}
        ]
        
        # Учитываем историю
        messages = []
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": content})
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logging.error(f"Image request error: {e}")
            return {"error": str(e)}