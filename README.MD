# 📡 API Интеграция: Что отправлять и что получать

### Установка библиотек питона

```bash
pip install -r requirements.txt

python main.py
```

## 📤 Отправка данных в API

### URL и метод:
```
POST http://localhost:8000/api/process-cv
```

### Headers:
```
Content-Type: application/json
```

### Body (что отправлять):
```json
{
  "vacancyPath": "/полный/путь/к/vacancy.docx",
  "cvPath": "/полный/путь/к/cv.docx"
}
```

### Примеры путей:
```json
{
  "vacancyPath": "/home/user/documents/vacancy.docx",
  "cvPath": "/home/user/documents/cv.docx"
}
```

или на Windows:
```json
{
  "vacancyPath": "C:\\Users\\user\\Documents\\vacancy.docx",
  "cvPath": "C:\\Users\\user\\Documents\\cv.docx"
}
```

## 📥 Что получаете в ответ

### Успешный ответ (200):
```json
{
  "score": 75.3,
  "verdict": "approved"
}
```

### Ошибка файла (404):
```json
{
  "detail": "Файл не найден: [описание ошибки]"
}
```

### Другая ошибка (500):
```json
{
  "detail": "Ошибка обработки: [описание ошибки]"
}
```

## 🌐 Примеры использования

### JavaScript Fetch:
```javascript
const requestData = {
  vacancyPath: "/home/user/vacancy.docx",
  cvPath: "/home/user/cv.docx"
};

fetch('http://localhost:8000/api/process-cv', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(requestData)
})
.then(response => response.json())
.then(data => {
  console.log('Score:', data.score);     // 75.3
  console.log('Verdict:', data.verdict); // "approved"
})
.catch(error => {
  console.error('Error:', error);
});
```

### curl пример:
```bash
curl -X POST http://localhost:8000/api/process-cv \
  -H "Content-Type: application/json" \
  -d '{
    "vacancyPath": "/home/user/vacancy.docx",
    "cvPath": "/home/user/cv.docx"
  }'
```

## 📋 Поля ответа

| Поле | Тип | Описание |
|------|-----|----------|
| `score` | float | Процент совпадения (0.0 - 100.0) |
| `verdict` | string | Результат: "approved" или "rejected" |

## ⚠️ Возможные ошибки

1. **404 Not Found**: Файлы не найдены
2. **422 Unprocessable Entity**: Неправильный формат запроса
3. **500 Internal Server Error**: Ошибка обработки

## 🎯 Логика verdict

```javascript
// На фронтенде вы можете использовать:
const verdict = score >= 50.0 ? 'approved' : 'rejected';
```

## 📝 Важные моменты

- Пути должны быть **абсолютными** (полными)
- Файлы должны существовать на сервере
- Сервер Python должен иметь доступ к этим файлам
- API работает на порту 8000 по умолчанию