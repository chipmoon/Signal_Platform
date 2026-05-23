# 🚀 Deploy lên Hugging Face Spaces

## Tại sao chuyển sang HF Spaces?

| Vấn đề Streamlit Cloud | Giải pháp HF Spaces |
|------------------------|---------------------|
| Sleep sau 1 giờ không dùng | **Không sleep** (luôn chạy) |
| Rebuild lỗi do torch/deps nặng | Build ổn định hơn, timeout dài hơn |
| IP bị block bởi vnstock | IP khác, ít bị block hơn |

---

## 📋 Bước 1: Tạo GitHub Repository

```bash
# Tạo repo mới trên GitHub, sau đó:
cd d:\Python_VS\trading_system
git init
git add .
git commit -m "Initial commit — Trading Intelligence Platform"
git remote add origin https://github.com/YOUR_USERNAME/trading-intelligence-platform.git
git push -u origin main
```

> ⚠️ **QUAN TRỌNG**: File `.env` đã được thêm vào `.gitignore`. ĐỪNG bao giờ commit `.env`!

---

## 📋 Bước 2: Tạo Hugging Face Space

1. Vào [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **"Create new Space"**
3. Điền thông tin:
   - **Space name**: `trading-intelligence-platform`
   - **License**: MIT
   - **SDK**: **Streamlit** ← Quan trọng
   - **Visibility**: Public hoặc Private tùy ý
4. Click **"Create Space"**

---

## 📋 Bước 3: Link GitHub → HF Spaces

### Option A: Push trực tiếp lên HF Spaces (Khuyến nghị)
```bash
# Clone HF Space repo
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/trading-intelligence-platform

# Copy tất cả files vào đó
# (hoặc dùng HF Spaces → Files → Upload files)

# Push lên
cd trading-intelligence-platform
git add .
git commit -m "Deploy Trading Intelligence Platform"
git push
```

### Option B: Dùng HF UI Upload
1. Vào Space của bạn trên HF
2. Click tab **"Files"** 
3. Click **"Upload files"**
4. Drag & drop toàn bộ thư mục `trading_system/` (trừ venv/, .env)

---

## 📋 Bước 4: Cấu hình Secrets (API Keys)

1. Vào Space → **Settings** → **Repository secrets**
2. Thêm từng secret:

| Key | Value |
|-----|-------|
| `GOOGLE_API_KEY` | API key Gemini của bạn |
| `TELEGRAM_BOT_TOKEN` | (tùy chọn) |
| `TELEGRAM_CHAT_ID` | (tùy chọn) |

> Secrets được inject tự động vào môi trường — code sẽ đọc qua `os.getenv()`.

---

## 📋 Bước 5: Verify Build

Sau khi push, vào Space và check **"Logs"**:

```
✅ Dấu hiệu build thành công:
- "Installing requirements.txt"
- "Your app is running on port 7860"

❌ Nếu thấy lỗi:
- "ModuleNotFoundError: xxx" → thêm module vào requirements.txt
- "Build timeout" → requirements quá nặng, cần optimize thêm
```

---

## 🔧 Files đã được tạo/cập nhật

| File | Mô tả |
|------|-------|
| `README.md` | YAML frontmatter cho HF Spaces (SDK: streamlit) |
| `packages.txt` | System apt packages (libgomp1 cho LightGBM) |
| `requirements.txt` | Đã bỏ torch/transformers nặng, giữ core libs |
| `.streamlit/config.toml` | Port 7860, dark theme, headless mode |
| `.streamlit/secrets.toml.example` | Template secrets (không commit) |
| `.gitignore` | Bảo vệ .env và secrets.toml |

---

## ⚡ Quick Commands

```powershell
# Khởi tạo git và push
cd d:\Python_VS\trading_system
git init
git add README.md packages.txt requirements.txt .streamlit/ .gitignore streamlit_app.py views/ src/ docs/
git commit -m "feat: prepare for Hugging Face Spaces deployment"
```

---

## 🌐 URL sau khi deploy

```
https://huggingface.co/spaces/YOUR_USERNAME/trading-intelligence-platform
```
