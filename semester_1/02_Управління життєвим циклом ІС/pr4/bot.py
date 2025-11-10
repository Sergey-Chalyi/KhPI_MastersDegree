# bot.py
"""
Простой rule-based чат-бот для интернет-магазина сантехники на aiogram (v2).
Один файл: bot.py

Зависимости:
pip install aiogram==2.25.1

Запуск:
1) Создайте бота в @BotFather, получите токен.
2) Вставьте токен ниже.
3) python bot.py
"""

from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher import FSMContext
from aiogram.dispatcher.filters.state import State, StatesGroup
import logging

# ---------- Настройки ----------
API_TOKEN = "YOUR_BOT_TOKEN_HERE"  # <-- вставьте сюда токен

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)

# Простая in-memory "база" товаров
PRODUCTS = {
    "Унітаз CeramicPro": {"price": 2500, "category": "Унітази", "id": "p1"},
    "Раковина SlimWash": {"price": 1800, "category": "Раковини", "id": "p2"},
    "Ванна Ocean 170": {"price": 7200, "category": "Ванни", "id": "p3"},
    "Змішувач EasyMix": {"price": 950, "category": "Змішувачі", "id": "p4"},
}

CATEGORIES = ["Унітази", "Раковини", "Ванни", "Змішувачі"]

# Хранение корзин: user_id -> list of product names
CARTS = {}

# ---------- FSM состояния ----------
class OrderStates(StatesGroup):
    choosing_category = State()
    choosing_product = State()
    confirm_add = State()
    checkout_name = State()
    checkout_phone = State()
    checkout_address = State()
    choosing_payment = State()


# ---------- Утилиты ----------
def get_products_by_category(cat):
    return [name for name, p in PRODUCTS.items() if p["category"] == cat]

def price_of(name):
    return PRODUCTS[name]["price"]

def cart_total(user_id):
    cart = CARTS.get(user_id, [])
    return sum(price_of(n) for n in cart)


# ---------- Keyboards ----------
def main_menu_kb():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add("📚 Каталог", "ℹ️ Доставка/Оплата")
    kb.add("🛒 Моя корзина", "💬 Зв'язок з оператором")
    return kb

def categories_kb():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for c in CATEGORIES:
        kb.add(c)
    kb.add("Назад")
    return kb

def products_kb(products):
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for p in products:
        kb.add(p)
    kb.add("Назад")
    return kb

def yes_no_kb():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    kb.add("Так", "Ні")
    return kb

def payment_kb():
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    kb.add("Онлайн оплата", "Оплата при отриманні")
    kb.add("Назад")
    return kb


# ---------- Обработчики ----------
@dp.message_handler(commands=["start", "help"])
async def cmd_start(message: types.Message):
    user_id = message.from_user.id
    if user_id not in CARTS:
        CARTS[user_id] = []
    text = (
        "Вітаю! Я чат-бот магазину сантехніки. Я можу допомогти знайти товар, "
        "оформити замовлення та відповісти на питання.\n\n"
        "Оберіть дію з меню."
    )
    await message.answer(text, reply_markup=main_menu_kb())


@dp.message_handler(lambda msg: msg.text == "📚 Каталог")
async def cmd_catalog(message: types.Message):
    await OrderStates.choosing_category.set()
    await message.answer("Оберіть категорію товарів:", reply_markup=categories_kb())


@dp.message_handler(state=OrderStates.choosing_category)
async def state_choose_category(message: types.Message, state: FSMContext):
    text = message.text
    if text == "Назад":
        await state.finish()
        await message.answer("Повернення в головне меню.", reply_markup=main_menu_kb())
        return
    if text not in CATEGORIES:
        await message.answer("Будь ласка, оберіть категорію з клавіатури.", reply_markup=categories_kb())
        return
    products = get_products_by_category(text)
    if not products:
        await message.answer("У цій категорії поки немає товарів.", reply_markup=main_menu_kb())
        await state.finish()
        return
    await state.update_data(category=text)
    await OrderStates.next()  # choosing_product
    await message.answer(f"Товари в категорії *{text}*:", parse_mode="Markdown", reply_markup=products_kb(products))


@dp.message_handler(state=OrderStates.choosing_product)
async def state_choose_product(message: types.Message, state: FSMContext):
    text = message.text
    if text == "Назад":
        await state.finish()
        await message.answer("Повернення в головне меню.", reply_markup=main_menu_kb())
        return
    data = await state.get_data()
    category = data.get("category")
    products = get_products_by_category(category)
    if text not in products:
        await message.answer("Оберіть товар з переліку.", reply_markup=products_kb(products))
        return
    await state.update_data(product=text)
    price = price_of(text)
    await OrderStates.next()  # confirm_add
    await message.answer(f"Ви обрали: *{text}* — {price} грн.\nДодати в кошик?", parse_mode="Markdown", reply_markup=yes_no_kb())


@dp.message_handler(state=OrderStates.confirm_add)
async def state_confirm_add(message: types.Message, state: FSMContext):
    text = message.text
    user_id = message.from_user.id
    if text == "Так":
        data = await state.get_data()
        product = data.get("product")
        CARTS.setdefault(user_id, []).append(product)
        await message.answer(f"Товар *{product}* додано в кошик. Сума: {cart_total(user_id)} грн.", parse_mode="Markdown", reply_markup=main_menu_kb())
        await state.finish()
        return
    elif text == "Ні":
        await message.answer("Добре. Повернення в головне меню.", reply_markup=main_menu_kb())
        await state.finish()
        return
    else:
        await message.answer("Будь ласка, оберіть Так або Ні.", reply_markup=yes_no_kb())


@dp.message_handler(lambda msg: msg.text == "🛒 Моя корзина")
async def cmd_cart(message: types.Message):
    user_id = message.from_user.id
    cart = CARTS.get(user_id, [])
    if not cart:
        await message.answer("Ваша корзина порожня.", reply_markup=main_menu_kb())
        return
    text_lines = [f"🧾 Ваша корзина ({len(cart)}):"]
    for i, name in enumerate(cart, 1):
        text_lines.append(f"{i}. {name} — {price_of(name)} грн")
    text_lines.append(f"\nСума: {cart_total(user_id)} грн")
    text_lines.append("\nЩо зробити далі?")
    kb = types.ReplyKeyboardMarkup(resize_keyboard=True)
    kb.add("Оформити замовлення", "Очистити корзину")
    kb.add("Назад")
    await message.answer("\n".join(text_lines), reply_markup=kb)


@dp.message_handler(lambda msg: msg.text == "Очистити корзину")
async def cmd_clear_cart(message: types.Message):
    user_id = message.from_user.id
    CARTS[user_id] = []
    await message.answer("Корзина очищена.", reply_markup=main_menu_kb())


@dp.message_handler(lambda msg: msg.text == "Оформити замовлення")
async def cmd_checkout_start(message: types.Message):
    user_id = message.from_user.id
    cart = CARTS.get(user_id, [])
    if not cart:
        await message.answer("Корзина порожня. Додайте товари перед оформленням.", reply_markup=main_menu_kb())
        return
    await OrderStates.checkout_name.set()
    await message.answer("Введіть ваше повне ім'я для оформлення замовлення:", reply_markup=types.ReplyKeyboardRemove())


@dp.message_handler(state=OrderStates.checkout_name)
async def state_checkout_name(message: types.Message, state: FSMContext):
    name = message.text.strip()
    if len(name) < 2:
        await message.answer("Введіть коректне ім'я.")
        return
    await state.update_data(name=name)
    await OrderStates.next()
    await message.answer("Введіть телефон (наприклад, +380XXXXXXXXX):")


@dp.message_handler(state=OrderStates.checkout_phone)
async def state_checkout_phone(message: types.Message, state: FSMContext):
    phone = message.text.strip()
    # Прості валідні перевірки
    if len(phone) < 9 or not any(ch.isdigit() for ch in phone):
        await message.answer("Введіть коректний номер телефону.")
        return
    await state.update_data(phone=phone)
    await OrderStates.next()
    await message.answer("Введіть адресу доставки (місто, вулиця, номер):")


@dp.message_handler(state=OrderStates.checkout_address)
async def state_checkout_address(message: types.Message, state: FSMContext):
    address = message.text.strip()
    if len(address) < 5:
        await message.answer("Введіть коректну адресу.")
        return
    await state.update_data(address=address)
    await OrderStates.next()
    await message.answer("Оберіть спосіб оплати:", reply_markup=payment_kb())


@dp.message_handler(state=OrderStates.choosing_payment)
async def state_choose_payment(message: types.Message, state: FSMContext):
    text = message.text
    user_id = message.from_user.id
    if text == "Назад":
        await state.finish()
        await message.answer("Скасовано оформлення.", reply_markup=main_menu_kb())
        return
    if text not in ["Онлайн оплата", "Оплата при отриманні"]:
        await message.answer("Оберіть спосіб оплати з клавіатури.", reply_markup=payment_kb())
        return
    data = await state.get_data()
    name = data.get("name")
    phone = data.get("phone")
    address = data.get("address")
    cart = CARTS.get(user_id, [])
    total = cart_total(user_id)
    # Здесь обычно интеграция с платежной системой. У нас — имитация.
    order_id = f"ORD{user_id % 10000}"
    # Очистим корзину
    CARTS[user_id] = []
    await state.finish()
    await message.answer(
        f"✅ Замовлення оформлено!\n\nНомер: *{order_id}*\nІм'я: {name}\nТелефон: {phone}\nАдреса: {address}\nСума: {total} грн\nСпосіб оплати: {text}",
        parse_mode="Markdown",
        reply_markup=main_menu_kb()
    )


@dp.message_handler(lambda msg: msg.text == "ℹ️ Доставка/Оплата")
async def cmd_info(message: types.Message):
    text = (
        "Доставка:\n"
        "- Доставка по місту: від 2 днів\n"
        "- Доставка по Україні: від 3-7 днів\n\n"
        "Оплата:\n"
        "- Онлайн оплата карткою\n"
        "- Оплата при отриманні (нал/безнал)\n\n"
        "Якщо потрібна додаткова інформація, оберіть '💬 Зв'язок з оператором'."
    )
    await message.answer(text, reply_markup=main_menu_kb())


@dp.message_handler(lambda msg: msg.text == "💬 Зв'язок з оператором")
async def cmd_operator(message: types.Message):
    await message.answer("Ви будете переадресовані на оператора. Час очікування може бути від 5 до 15 хвилин.\n(У прототипі оператор — ім'я@example.com)", reply_markup=main_menu_kb())


# Фолбек: якщо повідомлення не розпізнано і не в стані FSM
@dp.message_handler()
async def fallback_handler(message: types.Message):
    text = message.text.lower()
    # Популярні ключові слова: швидкий простий парсер
    if any(w in text for w in ["унітаз", "раковина", "ванн", "змішувач"]):
        # попрохати обрати категорію
        await message.answer("Схоже, ви шукаєте товар. Оберіть категорію:", reply_markup=categories_kb())
        await OrderStates.choosing_category.set()
        return
    if any(w in text for w in ["корзина", "кошик", "замовлення"]):
        await cmd_cart(message)
        return
    # Якщо нічого не підходить — підказати меню
    await message.answer("Не зрозумів запит. Використайте меню нижче.", reply_markup=main_menu_kb())


# ---------- Запуск ----------
if __name__ == "__main__":
    print("Bot is starting...")
    executor.start_polling(dp, skip_updates=True)
