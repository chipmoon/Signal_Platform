# PLAN: Market Alpha Scanner (Hybrid Approach B+C)

## 📌 Context
- **Objective**: Xây dựng một Market Scanner (Alpha Finder) để quét toàn bộ danh mục cổ phiếu/tài sản mà hệ thống đang hỗ trợ.
- **Goals**:
  1. Phát hiện dấu hiệu thao túng (Manipulation), xu hướng Bull/Bear.
  2. Lấy các thông số AI Forecast (Target Price, Confidence, Bias).
  3. Xuất kết quả ra file Excel để có cái nhìn tổng quan toàn thị trường.
  4. (Hướng C): Tích hợp trực tiếp lên giao diện Streamlit hiện tại để quét và hiển thị trực quan (Data Table) cùng với chức năng Export Excel.
- **Approach**: Hybrid B + C (Core Engine hỗ trợ đa luồng để tối ưu thời gian quét, và UI Component để hiển thị trên Streamlit).

---

## 🛠️ Task Breakdown

### Phase 1: Core Engine - Tuyến phòng thủ đa luồng (Option B)
Xây dựng module `src/scanner/market_scanner.py` (hoặc đặt trong `src/strategies/`).
- **Nhiệm vụ 1.1**: Định nghĩa danh sách các symbol mặc định cho các thị trường (VN30, Top US Tech, Commodities) hoặc cho phép user truyền vào danh sách tùy chọn (có thể dùng hàm tiện ích để lấy từ các plugins/US, VN).
- **Nhiệm vụ 1.2**: Viết logic `scan_single_asset(symbol)`:
  - Tải dữ liệu lịch sử ngắn hạn (VD: 200-300 ngày để đủ data cho Indicator/AI).
  - Tích hợp `VolumePriceDetector`, `BankParticipationMonitor`, `COTMonitor`.
  - Khởi tạo `AIPredictor`, thực hiện `predict` cho nến hiện tại (T+1).
  - Trích xuất metrics: Volume Signal, Bank Signal, COT Signal, AI Bias, AI Target Price, AI Confidence, Volatility Proxy.
- **Nhiệm vụ 1.3**: Viết function `run_market_scan(symbols)` sử dụng `concurrent.futures.ThreadPoolExecutor` để chạy song song `scan_single_asset` trên nhiều mã cùng lúc, giúp tăng tốc độ đáng kể.
- **Nhiệm vụ 1.4**: Tổng hợp kết quả thành `pandas.DataFrame` và cấu hình lưu ra file Excel (`Market_Alpha_Report_YYYYMMDD.xlsx`).

### Phase 2: Streamlit UI Integration - Trực quan hóa dữ liệu (Option C)
Tích hợp vào `streamlit_app.py` (hoặc tạo file view mới `views/scanner_view.py`).
- **Nhiệm vụ 2.1**: Tạo thêm một tab/phần mới tên là "🔍 Alpha Scanner".
- **Nhiệm vụ 2.2**: UI cho phép chọn Market (VN30, US Tech, Commodities, Cả ba,...) hoặc nhập danh sách comma-separated.
- **Nhiệm vụ 2.3**: Nút `[🚀 Run Alpha Scan]`. Khi bấm:
  - Hiển thị Progress Bar / Status Text (e.g., "Scanning 1/30...").
  - Chạy hàm `run_market_scan` từ Phase 1.
- **Nhiệm vụ 2.4**: Hiển thị bảng kết quả (`st.dataframe` hoặc `AGGrid`) kết hợp định dạng màu sắc:
  - Xanh (Bullish/High Confidence/Manipulation Up)
  - Đỏ (Bearish/Low Confidence/Manipulation Down)
- **Nhiệm vụ 2.5**: Cung cấp nút tải xuống (Download Button) kết quả dưới định dạng CSV/Excel.

### Phase 3: Testing & Refinement
- **Nhiệm vụ 3.1**: Xử lý exception và rate limits. Nếu một mã bị lỗi API (ví dụ Yahoo chặn), chỉ log warning báo lỗi mã đó thay vì crash toàn bộ tiến trình quét.
- **Nhiệm vụ 3.2**: Tối ưu hóa memory nếu cache data có sẵn.

---

## 🤖 Agent Assignments
- **`backend-specialist` / `python-expert`**: Chịu trách nhiệm Phase 1 (Core Engine, ThreadPoolExecutor, Pandas Data manipulation, Excel export).
- **`frontend-specialist` / `streamlit-expert`**: Chịu trách nhiệm Phase 2 (UI/UX Component, Progress Rendering, Color formatting).

---

## ✅ Verification Checklist
- [ ] Hàm `scan_single_asset` trả về chính xác một dict/Series các chỉ số cần thiết và không bị crash nếu thiếu data.
- [ ] Quét đa luồng 1 lúc n-mã (vd: VN30) tốn ít thời gian hơn cách quét tuần tự.
- [ ] File Excel được sinh ra với đầy đủ các cột: Symbol, Trend (Bias), Target Price, Confidence, Vol / Bank / COT Manipulation flags.
- [ ] UI trên Streamlit hoạt động trơn tru: có báo progress, hiển thị bảng màu rõ ràng.
- [ ] Bấm tải Excel từ Streamlit hoạt động tốt.

---
*Created by: project-planner agent*
