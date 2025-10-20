# NooLogicTask — мінімальний набір для Git

Цей пакет додає у репозиторій три файли, щоб не злити ключі та виглядати як люди.

## Що всередині
- **.gitignore** — ховає `.env`, кеші, IDE-файли, секрети.
- **.env.example** — публічний шаблон змінних оточення **без** ключів.
- **README.md** — коротка інструкція запуску.

## Як використати
1. Розпакуй файли у корінь проєкту (де лежить `AI Agent.py`).
2. Заповни локальний `.env` (НЕ коміть його).
3. Додай і пушни:
   ```bash
   git add .
   git commit -m "init: базовий git-пак без секретів"
   git branch -M main
   git remote add origin https://github.com/SMFqq/NooLogicTask.git
   git push -u origin main
   ```

## Секрети
- Справжні ключі тримай локально у `.env` або додай у GitHub → Settings → Secrets.
- Якщо є `service-account.json`, не клади його у репозиторій. Заший у `GOOGLE_SA_JSON_B64`.

Успіхів і менше продакшен-відкатів.
