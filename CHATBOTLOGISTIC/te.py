import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
from graph import build_graph
from chatstate import ChatState

# Khởi tạo LangGraph
graph_app = build_graph()

# Lệnh /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Chào bạn. Vui lòng nhập thông tin đơn hàng để hệ thống xử lý và đưa ra dự đoán phương thức vận chuyển."
    )

# Lệnh /help
async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Gửi thông tin đơn hàng bao gồm nơi gửi, nơi nhận, trọng lượng, chi phí, vendor và phương thức mong muốn.\n"
        "Hệ thống sẽ xử lý và đưa ra gợi ý phù hợp."
    )

# ✅ Lệnh /reset
async def reset_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    context.chat_data.clear()  # 💣 Xoá hết state hiện tại

    await context.bot.send_message(
        chat_id=chat_id,
        text="🔄 Cuộc trò chuyện đã được đặt lại. Vui lòng nhập thông tin đơn hàng mới để tiếp tục."
    )

# Xử lý tin nhắn văn bản
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    user_text = update.message.text

    # Tạo hoặc khôi phục trạng thái
    state = ChatState(user_id=str(chat_id), input_text=user_text)

    # ✅ Nếu user đang trong support mode
    if hasattr(context.chat_data, "support_mode") and context.chat_data["support_mode"]:
        context.chat_data["support_count"] += 1

        # Xử lý tiếp nội dung hỗ trợ (đơn giản: echo lại hoặc đưa vào GPT tùy cấp)
        if context.chat_data["support_count"] < 5:
            await context.bot.send_message(
                chat_id=chat_id,
                text=f"🤖 Tôi đã ghi nhận yêu cầu hỗ trợ: \"{user_text}\". Quý khách còn {5 - context.chat_data['support_count']} lượt tư vấn miễn phí."
            )
            return
        else:
            context.chat_data["support_mode"] = False
            context.chat_data["support_count"] = 0
            await context.bot.send_message(
                chat_id=chat_id,
                text="🛑 Đã kết thúc 5 lượt tư vấn bổ sung. Nếu cần hỏi lại, vui lòng nhập thông tin đơn hàng mới hoặc dùng lệnh /reset."
            )
            return

    # ✅ Xử lý tin nhắn bình thường
    result = graph_app.invoke(state)
    final_answer = result.get("final_answer", "Hệ thống không thể xử lý thông tin vừa nhận.")

    # Lưu trạng thái support nếu có
    if getattr(result, "support_mode", False):
        context.chat_data["support_mode"] = True
        context.chat_data["support_count"] = 0

    await context.bot.send_message(chat_id=chat_id, text=final_answer)


# Chạy bot
if __name__ == "__main__":
    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("Chưa thiết lập biến môi trường TELEGRAM_TOKEN")

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("reset", reset_cmd))  # ✅ Thêm handler cho /reset
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot đang chạy bằng polling...")
    app.run_polling()
