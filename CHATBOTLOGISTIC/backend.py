import os
import re
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters
)
from graph import build_graph
from chatstate import ChatState
from state_store import reset_user_state

# ===== Khởi tạo LangGraph =====
graph_app = build_graph()

# ===== Store session state per user =====
user_states = {}

# ===== Clean Markdown safely =====
def clean_markdown(text: str) -> str:
    text = text.replace("```", "")
    text = re.sub(r"[*_`]", "", text)
    text = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1 (\2)", text)
    text = text.replace("&", "and")
    return text

# ===== /start =====
async def start(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚚 *Welcome to the Smart Logistics Assistant!*\n\n"
        "Just send me your shipment info and I’ll suggest the best *transport method*.",
        parse_mode=ParseMode.MARKDOWN
    )

# ===== /help =====
async def help_cmd(update: Update, _: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📦 *How to use:*\n\n"
        "➤ Destination country\n"
        "➤ Vendor name\n"
        "➤ Weight, freight cost, pack price\n"
        "➤ Delivery expectation (e.g. in 5 days)\n\n"
        "I’ll respond with the best *shipment method* and reasons why.",
        parse_mode=ParseMode.MARKDOWN
    )

# ===== /reset =====
async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = str(update.effective_chat.id)
    context.chat_data.clear()
    reset_user_state(user_id)
    user_states.pop(user_id, None)
    await update.message.reply_text(
        "🔄 *Reset done.* Please send a new shipment request.",
        parse_mode=ParseMode.MARKDOWN
    )

# ===== Handle user message =====
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_id = str(chat_id)

    if not update.message or not update.message.text:
        await update.message.reply_text("⚠️ Please send a text message.")
        return

    user_text = update.message.text.strip()
    print("=" * 60)
    print(f"📨 [User {chat_id}] Input: {user_text}")

    # 👉 Lấy state cũ nếu có
    prev_state = user_states.get(user_id)
    state = ChatState(user_id=user_id, input_text=user_text)

    if isinstance(prev_state, ChatState):
        state.support_count = prev_state.support_count + 1
        state.message_count = prev_state.message_count + 1
        state.support_mode = True
    else:
        state.support_count = 0
        state.message_count = 1
        state.support_mode = False

    print(f"📊 message_count: {state.message_count} | support_count: {state.support_count}")

    try:
        result = graph_app.invoke(state)

        # ✅ Convert dict → ChatState nếu cần
        if isinstance(result, dict):
            result_state = ChatState(**result)
        else:
            result_state = result

        user_states[user_id] = result_state
    except Exception as e:
        print("❌ LangGraph error:", e)
        await update.message.reply_text(
            f"💥 *Internal error:* `{str(e)}`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    # === Chuẩn hóa lấy final_answer ===
    raw_answer = result.get("final_answer", "").strip()
    final_answer = clean_markdown(raw_answer)

    if not final_answer:
        final_answer = "⚠️ The system couldn't process your request. Please try again later."

    print("✅ Final answer to be sent:\n", final_answer)

    try:
        await update.message.reply_text(final_answer, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        print(f"❌ Telegram send error: {e}")
        await update.message.reply_text(
            "⚠️ Telegram could not send the message properly. Try again or /reset.",
            parse_mode=None
        )

# ===== Main =====
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("❌ TELEGRAM_TOKEN is missing.")

    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🚀 Telegram bot running with Markdown enabled.")
    app.run_polling()
