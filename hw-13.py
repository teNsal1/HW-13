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

class ChatFacade:
    """Фасад для работы с Mistral API с поддержкой стратегий"""
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.text_strategy = TextRequestStrategy(api_key)
        self.image_strategy = ImageRequestStrategy(api_key)
        self.current_strategy = self.text_strategy  # Стратегия по умолчанию
        self.models = {
            "text": ["mistral-large-latest", "mistral-small-latest"],
            "image": ["pixtral-12b-2409"]
        }

    def change_strategy(self, strategy_type: str) -> None:
        """
        Изменяет текущую стратегию выполнения запросов
        
        Args:
            strategy_type: Тип стратегии ('text' или 'image')
            
        Raises:
            ValueError: Если указан неизвестный тип стратегии
        """
        if strategy_type == "text":
            self.current_strategy = self.text_strategy
        elif strategy_type == "image":
            self.current_strategy = self.image_strategy
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    def select_model(self) -> str:
        """Выбирает модель, соответствующую текущей стратегии"""
        strategy_type = "text" if self.current_strategy == self.text_strategy else "image"
        available_models = self.models.get(strategy_type, [])
        
        if not available_models:
            raise ValueError("No models available for current strategy")
            
        print(f"Available models for {strategy_type} strategy:")
        for i, model in enumerate(available_models, 1):
            print(f"{i}. {model}")
            
        while True:
            try:
                choice = int(input("Select model number: "))
                if 1 <= choice <= len(available_models):
                    return available_models[choice-1]
                print("Invalid choice, try again")
            except ValueError:
                print("Please enter a number")

    def ask_question(self, text: str, model: str, image_path: Optional[str] = None) -> dict:
        """
        Выполняет запрос с использованием текущей стратегии
        
        Args:
            text: Текст запроса
            model: Используемая модель
            image_path: Путь к изображению (для стратегии изображений)
            
        Returns:
            Ответ API
        """
        # Пока без истории
        response = self.current_strategy.execute(
            text=text,
            model=model,
            history=None,
            image_path=image_path
        )
        return response


if __name__ == "__main__":
    api_key = os.getenv("MISTRAL_API_KEY", "")
    chat = ChatFacade(api_key)
    
    # Работа с текстовой стратегией
    chat.change_strategy("text")
    model = chat.select_model()
    response = chat.ask_question("Расскажи о последних новостях в IT", model)
    print("Text response:", response)
    
    # Переключение на стратегию с изображениями
    chat.change_strategy("image")
    model = chat.select_model()
    response = chat.ask_question(
        "Что изображено на картинке?",
        model,
        image_path="path/to/image.jpg"
    )
    print("Image response:", response)