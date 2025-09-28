# Звіт: лабораторна робота №4

**Дисципліна:** Управління проєктами розробки інтелектуальних систем

**Тема:** Моделювання предметної області у вигляді фреймів

**Студент:** Чалий Сергій (КН-Н425, 13 в списку групи)

**Варіант:** 20

---

### Приклади фреймів (у форматі JSON-подібному для ясності)

**Frame: User**

```
User {
  id: UUID
  name: string
  email: string
  currency: enum {UAH, USD, EUR}
  accounts: list<Account>
  preferences: {language, notification_settings}
}
```

**Frame: Account**

```
Account {
  id: UUID
  owner: User
  type: enum {Cash, Card, Bank}
  balance: decimal
  transactions: list<Transaction>
}
```

**Frame: Transaction**

```
Transaction {
  id: UUID
  date: datetime
  amount: decimal
  currency: enum
  account: Account
  category: Category
  description: text
  tags: list<string>
  recurrent: boolean
  source: enum {manual, import, bank}
}
```

**Frame: Budget**

```
Budget {
  id: UUID
  owner: User
  period: {start_date, end_date}
  category_limits: map<Category, decimal>
  alert_threshold: percent
}
```

**Frame: Rule (Autocategorization)**

```
Rule {
  id: UUID
  pattern: string (regex / merchant)
  category: Category
  priority: int
  active: bool
}
```

