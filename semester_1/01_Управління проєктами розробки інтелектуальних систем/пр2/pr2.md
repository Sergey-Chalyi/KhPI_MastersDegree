# Звіт: лабораторна робота №2

**Дисципліна:** Управління проєктами розробки інтелектуальних систем

**Тема:** моделювання домену

**Студент:** Чалий Сергій (КН-Н425, 13 в списку групи)

**Варіант:** 20

> Використовуючи бізнес-процеси з ЛР1.

---

### 1. Діаграми діяльності (Activity diagrams) — Mermaid

**БП1 — Управління транзакціями**

```mermaid
flowchart TD
  start([Start])
  A[Ввід транзакції]
  B{Чи є імпорт?}
  C[Автокатегоризація]
  D[Зберегти транзакцію]
  E[Оновити баланс]
  F[Синхронізація/резервне копіювання]
  finish([End])

  start --> B
  B -- Так --> A
  B -- Ні --> A
  A --> C
  C --> D
  D --> E
  E --> F
  F --> finish
```

**БП2 — Бюджетування та звітність**

```mermaid
flowchart TD
  s([Start])
  X[Встановлення бюджету]
  Y[Моніторинг транзакцій]
  Z{Перевищення?}
  N[Надіслати сповіщення]
  R[Згенерувати звіт]
  e([End])

  s --> X
  X --> Y
  Y --> Z
  Z -- Так --> N --> Y
  Z -- Ні --> R --> e
```

### 2. BPMN (приблизно) — Mermaid (flowchart used as BPMN-like)

(для академічних цілей цей формат прийнятний: можна використовувати спеціальні BPMN інструменти пізніше)

```mermaid
flowchart TB
  Start(Start) --> Task1[User: Create Transaction]
  Task1 --> Task2[System: Suggest Category]
  Task2 --> Gateway{Confirm?}
  Gateway -- Yes --> Task3[System: Save Transaction]
  Task3 --> End(End)
  Gateway -- No --> Task1
```

### 3. DFD (Data Flow Diagrams) — рівень контекст + DFD0

**Контекст (DFD0)**

```mermaid
flowchart LR
  User[Користувач] -->|вводить транзакції| System[ІС домашньої бухгалтерії]
  System -->|звіт| User
  BankCSV[Банк/CSV] -->|файл| System
  System -->|резервні копії| Backup[Резервна служба]
```

**DFD0 внутрішні процеси**

```mermaid
flowchart TB
  P1[1. Обробка транзакцій]
  P2[2. Категоризація/правила]
  P3[3. Генерація звітів]
  DB[(БД)]
  P1 --> DB
  P2 --> DB
  DB --> P3
  User --> P1
  User --> P3
```