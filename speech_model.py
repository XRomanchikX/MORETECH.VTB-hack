import json
import time
import re
import os
from typing import Dict, List, Tuple
from openai import OpenAI
from gtts import gTTS
import pygame
from io import BytesIO
import vosk
import sounddevice as sd
import queue
import numpy as np
from datetime import datetime

class AdvancedVoiceInterviewSystem:
    def __init__(self, resume_data: Dict):
        # Основной клиент для генерации вопросов
        self.client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        # Отдельный клиент для оценки (можно использовать ту же модель, но логика разделена)
        self.evaluation_client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        
        self.generation_model = "meta-llama-3-8b-instruct"  # Для генерации вопросов
        self.evaluation_model = "qwen2-vl-7b-instruct"      # Для оценки ответов
        
        self.resume_data = resume_data
        pygame.mixer.init()
        
        # Инициализация Vosk
        self.setup_voice_recognition()
        
        self.interview_questions = []
        self.ideal_answers = {}
        self.candidate_answers = []
        self.scores = []
        self.current_question_index = 0
        self.is_generating_next = False
        self.silence_timeout = 4.0  # Уменьшено до 2 секунд для более быстрой реакции
        self.silence_threshold = 0.01  # Порог уровня звука для детектирования тишины
        
        # Память диалога для адаптивных вопросов
        self.dialog_memory = []
        self.conversation_history = []
        self.follow_up_questions = []
        self.candidate_introduction = ""  # Для хранения рассказа кандидата о себе

    def setup_voice_recognition(self):
        """Инициализация распознавания речи Vosk с улучшенными настройками"""
        try:
            model_path = "vosk-model-small-ru-0.22"
            if not os.path.exists(model_path):
                model_path = "vosk-model-ru"
            
            self.vosk_model = vosk.Model(model_path)
            self.sample_rate = 16000
            self.audio_queue = queue.Queue()
            self.recognition_active = False
            print("✅ Голосовое распознавание инициализировано")
        except Exception as e:
            print(f"❌ Ошибка инициализации Vosk: {e}")
            self.vosk_model = None

    def listen_to_speech_with_silence_detection(self, timeout: int = 120) -> Tuple[str, float]:
        """
        Слушает речь с детектированием тишины
        Возвращает (распознанный текст, длительность ответа)
        """
        if not self.vosk_model:
            return self.fallback_listen(), 0.0
            
        print("🎤 Можете говорить")
        
        try:
            rec = vosk.KaldiRecognizer(self.vosk_model, self.sample_rate)
            full_text = ""
            last_speech_time = time.time()
            start_time = time.time()
            speech_detected = False
            
            def audio_callback(indata, frames, time_info, status):
                if status:
                    print(status)
                
                # Детектирование тишины
                audio_data = bytes(indata)
                audio_level = np.sqrt(np.mean(np.frombuffer(audio_data, dtype=np.int16)**2))
                
                if audio_level > self.silence_threshold * 32768:  # 32768 - максимальное значение для int16
                    nonlocal last_speech_time
                    last_speech_time = time.time()
                    nonlocal speech_detected
                    speech_detected = True
                
                self.audio_queue.put((audio_data, time.time()))
            
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=8000,
                dtype='int16',
                channels=1,
                callback=audio_callback
            ):
                self.recognition_active = True
                
                while time.time() - start_time < timeout and self.recognition_active:
                    try:
                        data, timestamp = self.audio_queue.get(timeout=0.5)
                        
                        if rec.AcceptWaveform(data):
                            result = json.loads(rec.Result())
                            if result.get('text'):
                                full_text += " " + result['text']
                                print(f"Распознано: {result['text']}")
                                speech_detected = True
                        
                        # Проверяем тишину - только если уже была речь
                        if speech_detected and time.time() - last_speech_time > self.silence_timeout:
                            print("⏸️  Обнаружена пауза, завершаем запись...")
                            break
                            
                    except queue.Empty:
                        # Проверяем тишину в случае пустой очереди
                        if speech_detected and time.time() - last_speech_time > self.silence_timeout:
                            print("⏸️  Обнаружена пауза, завершаем запись...")
                            break
                        continue
                
                # Получаем финальный результат
                final_result = json.loads(rec.FinalResult())
                if final_result.get('text'):
                    full_text += " " + final_result['text']
                
            duration = time.time() - start_time
            return full_text.strip() if full_text.strip() else "Кандидат не ответил", duration
            
        except Exception as e:
            print(f"Ошибка распознавания речи: {e}")
            return self.fallback_listen(), 0.0
        finally:
            self.recognition_active = False

    def fallback_listen(self) -> str:
        """Резервный метод ввода через текст"""
        print("🎤 Для голосового ввода установите Vosk. Сейчас используйте текстовый ввод.")
        try:
            return input("💬 Ваш ответ (текст): ")
        except:
            return "Кандидат не ответил"

    def stop_listening(self):
        """Останавливает прослушивание"""
        self.recognition_active = False

    def evaluate_answer_and_decide(self, question: str, candidate_answer: str, ideal_answer: str) -> Tuple[float, str, bool]:
        """
        Оценивает ответ и принимает решение: задавать доп. вопрос или переходить дальше
        Возвращает (оценка, фидбек, нужно_ли_углубляться)
        """
        evaluation_prompt = f"""
        Ты - эксперт по оценке ответов на собеседованиях. Оцени ответ кандидата и прими решение:
        нужно ли задавать дополнительный уточняющий вопрос по этой теме или можно переходить к следующей.
        
        ВОПРОС: {question}
        
        ОТВЕТ КАНДИДАТА: {candidate_answer}
        
        ИДЕАЛЬНЫЙ ОТВЕТ: {ideal_answer}
        
        Проанализируй ответ по критериям:
        1. Полнота и глубина ответа
        2. Конкретность примеров
        3. Соответствие вопросу
        4. Ясность изложения
        
        Если ответ поверхностный, неконкретный или требует уточнения - верни решение углубиться.
        Если ответ полный, конкретный и исчерпывающий - верни решение двигаться дальше.
        
        Верни ответ в формате JSON:
        {{
            "score": число от 1.0 до 5.0,
            "feedback": "конструктивный фидбек на русском",
            "need_follow_up": true/false (нужен ли дополнительный вопрос),
            "reason": "причина решения"
        }}
        """
        
        try:
            print("🤖 Оцениваю ответ и принимаю решение...")
            
            response = self.evaluation_client.chat.completions.create(
                model=self.evaluation_model,
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
                max_tokens=400,
                timeout=45
            )
            
            if response and response.choices:
                result_text = response.choices[0].message.content
                
                # Пытаемся извлечь JSON из ответа
                try:
                    json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                    if json_match:
                        result_data = json.loads(json_match.group())
                        score = float(result_data.get('score', 3.0))
                        feedback = result_data.get('feedback', 'Автоматическая оценка')
                        need_follow_up = bool(result_data.get('need_follow_up', True))
                        return max(1.0, min(5.0, score)), feedback, need_follow_up
                except json.JSONDecodeError:
                    pass
            
            # Fallback - упрощенная оценка
            return self.simple_evaluation(candidate_answer), "Автоматическая оценка", len(candidate_answer) < 100
            
        except Exception as e:
            print(f"⚠️  Ошибка оценки: {e}")
            return self.simple_evaluation(candidate_answer), "Ошибка оценки", len(candidate_answer) < 100

    def simple_evaluation(self, candidate_answer: str) -> float:
        """Упрощенная оценка на основе эвристик"""
        length = len(candidate_answer)
        if length < 30:
            return 2.0
        elif length < 100:
            return 3.0
        elif length < 200:
            return 4.0
        else:
            return 4.5

    def generate_follow_up_question(self, question: str, answer: str) -> str:
        """Генерирует уточняющий вопрос на основе ответа"""
        prompt = f"""
        На основе предыдущего вопроса и ответа кандидата, сгенерируй уточняющий вопрос.
        
        ПРЕДЫДУЩИЙ ВОПРОС: {question}
        ОТВЕТ КАНДИДАТА: {answer}
        
        Сгенерируй один уточняющий вопрос на русском языке, который:
        1. Углубляется в детали предыдущего ответа
        2. Просит привести конкретные примеры или цифры
        3. Помогает лучше понять компетенции кандидата
        4. Связан с позицией в ВТБ
        
        Верни только текст вопроса.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100,
                timeout=30
            )
            
            if response and response.choices:
                question = response.choices[0].message.content.strip()
                return question
            else:
                return "Можете рассказать об этом подробнее?"
                
        except Exception as e:
            print(f"⚠️  Ошибка генерации уточняющего вопроса: {e}")
            return "Можете рассказать об этом подробнее?"

    def generate_question_based_on_introduction(self, introduction: str) -> str:
        """Генерирует первый вопрос на основе рассказа кандидата о себе"""
        prompt = f"""
        На основе рассказа кандидата о себе, сгенерируй первый вопрос для собеседования.
        
        РАССКАЗ КАНДИДАТА: {introduction}
        
        ДАННЫЕ ИЗ РЕЗЮМЕ:
        - Опыт работы: {self.resume_data.get('experience', 'не указан')}
        - Должность: {self.resume_data.get('position', 'не указана')}
        - Навыки: {', '.join(self.resume_data.get('skills', []))}
        
        Сгенерируй релевантный вопрос на русском языке, который:
        1. Связан с опытом кандидата
        2. Касается его мотивации работать в ВТБ
        3. Исследует ключевые компетенции
        4. Является естественным продолжением его рассказа
        
        Верни только текст вопроса.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150,
                timeout=30
            )
            
            if response and response.choices:
                question = response.choices[0].message.content.strip()
                return question
            else:
                return "Расскажите о вашем профессиональном опыте и ключевых достижениях?"
                
        except Exception as e:
            print(f"⚠️  Ошибка генерации вопроса: {e}")
            return "Расскажите о вашем профессиональном опыте и ключевых достижениях?"

    def generate_next_question_based_on_history(self, conversation_history: List[Dict]) -> str:
        """Генерирует следующий вопрос на основе истории диалога"""
        # Формируем историю диалога для контекста
        history_text = ""
        for i, msg in enumerate(conversation_history[-6:]):  # Берем последние 6 сообщений
            role = "Интервьюер" if msg['role'] == 'interviewer' else "Кандидат"
            history_text += f"{role}: {msg['content']}\n"
        
        prompt = f"""
        На основе истории диалога на собеседовании, сгенеририуй следующий вопрос.
        
        ИСТОРИЯ ДИАЛОГА:
        {history_text}
        
        ДАННЫЕ КАНДИДАТА:
        - Опыт: {self.resume_data.get('experience', 'не указан')}
        - Позиция: {self.resume_data.get('position', 'не указана')}
        - Навыки: {', '.join(self.resume_data.get('skills', []))}
        
        Сгенерируй релевантный вопрос на русском языке, который:
        1. Является логическим продолжением беседы
        2. Исследует новые аспекты компетенций кандидата
        3. Связан с работой в ВТБ
        4. Учитывает предыдущие ответы кандидата
        
        Верни только текст вопроса.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150,
                timeout=30
            )
            
            if response and response.choices:
                question = response.choices[0].message.content.strip()
                return question
            else:
                return "Расскажите о вашем опыте решения сложных профессиональных задач?"
                
        except Exception as e:
            print(f"⚠️  Ошибка генерации вопроса: {e}")
            return "Расскажите о вашем опыте решения сложных профессиональных задач?"

    def text_to_speech(self, text: str, language: str = 'ru'):
        """Преобразует текст в речь"""
        try:
            tts = gTTS(text=text, lang=language, slow=False)
            audio_bytes = BytesIO()
            tts.write_to_fp(audio_bytes)
            audio_bytes.seek(0)
            
            pygame.mixer.music.load(audio_bytes)
            pygame.mixer.music.play()
            
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
                
        except Exception as e:
            print(f"Ошибка синтеза речи: {e}")

    def get_ideal_answer(self, question: str) -> str:
        """Получает идеальный ответ от LLM"""
        prompt = f"""
        Как опытный специалист, дайте идеальный ответ на вопрос собеседования в ВТБ.
        
        Вопрос: {question}
        
        Учитывайте что кандидат:
        - {self.resume_data.get('experience', 'Имеет опыт работы')}
        - Претендует на позицию: {self.resume_data.get('position', 'специалиста')}
        - Навыки: {', '.join(self.resume_data.get('skills', []))}
        
        Ответ должен быть профессиональным, содержать конкретные примеры.
        Объем: 2-3 предложения.
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.generation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=150,
                timeout=30
            )
            
            if response and response.choices:
                return response.choices[0].message.content.strip()
            else:
                return "Идеальный ответ не сгенерирован"
                
        except Exception as e:
            print(f"⚠️  Ошибка генерации ответа: {e}")
            return "Идеальный ответ не сгенерирован"

    def analyze_candidate_fit(self) -> Dict:
        """Анализирует соответствие кандидата позиции"""
        if not self.candidate_answers:
            return {"score": 0, "recommendation": "Недостаточно данных"}
        
        total_score = sum(self.scores)
        avg_score = total_score / len(self.scores)
        
        # Анализ длительности ответов
        total_duration = sum(ans.get('duration_seconds', 0) for ans in self.candidate_answers)
        avg_duration = total_duration / len(self.candidate_answers)
        
        # Анализ ключевых слов в ответах
        all_answers = " ".join([ans['answer'] for ans in self.candidate_answers])
        keywords = ['опыт', 'проект', 'решил', 'настроил', 'команда', 'технологи', 'развитие', 'ВТБ', 'банк']
        keyword_count = sum(1 for word in keywords if word.lower() in all_answers.lower())
        
        # Формирование рекомендации
        if avg_score >= 4.2 and avg_duration > 30 and keyword_count >= 5:
            recommendation = "Сильный кандидат, рекомендован к найму"
            hiring_status = "Рекомендован"
        elif avg_score >= 3.5:
            recommendation = "Подходящий кандидат, требуется дополнительное собеседование"
            hiring_status = "На рассмотрении"
        else:
            recommendation = "Не соответствует требованиям позиции"
            hiring_status = "Не рекомендован"
        
        return {
            "average_score": round(avg_score, 1),
            "total_duration_seconds": round(total_duration, 1),
            "average_duration_seconds": round(avg_duration, 1),
            "keyword_match_count": keyword_count,
            "recommendation": recommendation,
            "hiring_status": hiring_status,
            "interview_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

    def conduct_interview(self):
        """Проводит адаптивное собеседование"""
        print("=" * 60)
        print("🎙️  АДАПТИВНАЯ СИСТЕМА ГОЛОСОВОГО СОБЕСЕДОВАНИЯ ВТБ")
        print("=" * 60)
        
        # Приветствие и представление
        greeting = "Добро пожаловать на собеседование в ВТБ! Меня зовут Виктория, я ваша виртуальная помощница. Давайте начнем с знакомства. Пожалуйста, расскажите немного о себе: о вашем опыте, навыках и почему вы заинтересованы в работе в ВТБ."
        print(f"👋 {greeting}")
        self.text_to_speech(greeting)
        
        # Слушаем рассказ кандидата о себе
        print("\n🎤 Слушаем рассказ кандидата о себе...")
        introduction, intro_duration = self.listen_to_speech_with_silence_detection(timeout=180)
        self.candidate_introduction = introduction
        print(f"💬 Кандидат: {introduction}")
        
        # Сохраняем в историю
        self.conversation_history.append({
            'role': 'interviewer',
            'content': greeting,
            'timestamp': time.time()
        })
        self.conversation_history.append({
            'role': 'candidate', 
            'content': introduction,
            'timestamp': time.time()
        })
        
        # Генерируем первый вопрос на основе рассказа
        print("🤖 Генерирую первый вопрос на основе рассказа кандидата...")
        first_question = self.generate_question_based_on_introduction(introduction)
        ideal_answer = self.get_ideal_answer(first_question)
        
        print(f"\n👩‍💼 HR: {first_question}")
        self.text_to_speech(first_question)
        
        # Основной цикл собеседования
        question_count = 0
        max_questions = 8
        
        while question_count < max_questions:
            # Слушаем ответ
            candidate_answer, answer_duration = self.listen_to_speech_with_silence_detection(timeout=180)
            print(f"💬 Ответ: {candidate_answer}")
            
            # Сохраняем ответ
            self.candidate_answers.append({
                'question': first_question,
                'answer': candidate_answer,
                'duration_seconds': round(answer_duration, 1),
                'ideal_answer': ideal_answer,
                'question_type': 'основной'
            })
            
            # Добавляем в историю
            self.conversation_history.append({
                'role': 'interviewer',
                'content': first_question,
                'timestamp': time.time()
            })
            self.conversation_history.append({
                'role': 'candidate', 
                'content': candidate_answer,
                'timestamp': time.time()
            })
            
            # Оцениваем ответ и принимаем решение
            score, feedback, need_follow_up = self.evaluate_answer_and_decide(
                first_question, candidate_answer, ideal_answer
            )
            
            self.scores.append(score)
            
            print(f"⭐ Оценка: {score}/5")
            print(f"📝 Фидбек: {feedback}")
            print(f"🔍 Нужен доп. вопрос: {'Да' if need_follow_up else 'Нет'}")
            
            # Если нужен дополнительный вопрос
            if need_follow_up and question_count < max_questions - 1:
                follow_up_question = self.generate_follow_up_question(first_question, candidate_answer)
                print(f"\n🔍 Уточняющий вопрос: {follow_up_question}")
                self.text_to_speech(follow_up_question)
                
                # Обновляем текущий вопрос для следующей итерации
                first_question = follow_up_question
                ideal_answer = self.get_ideal_answer(follow_up_question)
            else:
                # Генерируем следующий вопрос на основе истории
                if question_count < max_questions - 1:
                    next_question = self.generate_next_question_based_on_history(self.conversation_history)
                    print(f"\n📝 Следующий вопрос: {next_question}")
                    self.text_to_speech(next_question)
                    
                    first_question = next_question
                    ideal_answer = self.get_ideal_answer(next_question)
            
            question_count += 1
            
            # Небольшая пауза между вопросами
            time.sleep(1)
        
        # Завершение собеседования
        closing = "Спасибо за ваши ответы! На этом собеседование завершено. Результаты будут обработаны и направлены вам в ближайшее время."
        print(f"\n👋 {closing}")
        self.text_to_speech(closing)
        
        # Финальный анализ
        analysis = self.analyze_candidate_fit()
        print(f"\n{'='*60}")
        print("📊 ИТОГОВЫЙ АНАЛИЗ КАНДИДАТА")
        print("=" * 60)
        print(f"🎯 Средний балл: {analysis['average_score']}/5")
        print(f"⏱️  Общее время ответов: {analysis['total_duration_seconds']} сек")
        print(f"📈 Соответствие ключевым словам: {analysis['keyword_match_count']}")
        print(f"💼 Рекомендация: {analysis['recommendation']}")
        
        # Сохраняем результаты
        results = {
            'candidate_info': self.resume_data,
            'interview_results': analysis,
            'answers': self.candidate_answers,
            'scores': self.scores,
            'conversation_history': self.conversation_history
        }
        
        with open(f"interview_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print("💾 Результаты сохранены в файл")

    def run(self):
        """Запускает систему собеседования"""
        try:
            self.conduct_interview()
        except Exception as e:
            print(f"❌ Ошибка: {e}")
        finally:
            self.stop_listening()

# Пример данных резюме
resume_data = {
    "gender": "Мужской",
    "age": "35 лет",
    "experience": "10 лет в IT-инфраструктуре",
    "position": "Ведущий инженер IT-инфраструктуры",
    "skills": ["Windows Server", "Active Directory", "VMware", "Сетевое оборудование", "Linux", "Docker"]
}

if __name__ == "__main__":
    system = AdvancedVoiceInterviewSystem(resume_data)
    system.run()