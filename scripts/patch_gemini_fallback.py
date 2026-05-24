"""Patch ai_forecast.py to add Gemini fallback when Mozyfin credits exhausted."""
import re

file = "views/ai_forecast.py"
with open(file, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add Gemini import after Mozyfin import block
old_import = """# Mozyfin integration (optional - graceful fallback if key missing)
try:
    from src.mozyfin_client import MozyfinClient as _MozyfinClient
    _MOZYFIN_AVAILABLE = True
except ImportError:
    _MOZYFIN_AVAILABLE = False


def _get_mozyfin_client():
    \"\"\"Get Mozyfin client, returns None if API key not configured.\"\"\"
    if not _MOZYFIN_AVAILABLE:
        return None
    try:
        return _MozyfinClient()
    except Exception:
        return None"""

new_import = """# Mozyfin integration (optional - graceful fallback if key missing)
try:
    from src.mozyfin_client import MozyfinClient as _MozyfinClient
    _MOZYFIN_AVAILABLE = True
except ImportError:
    _MOZYFIN_AVAILABLE = False

# Gemini fallback (Google AI Studio - free 1500 req/day)
try:
    from src.gemini_client import GeminiClient as _GeminiClient
    _GEMINI_AVAILABLE = True
except ImportError:
    _GEMINI_AVAILABLE = False


def _get_mozyfin_client():
    \"\"\"Get Mozyfin client, returns None if API key not configured.\"\"\"
    if not _MOZYFIN_AVAILABLE:
        return None
    try:
        return _MozyfinClient()
    except Exception:
        return None


def _get_gemini_client():
    \"\"\"Get Gemini client, returns None if API key not configured.\"\"\"
    if not _GEMINI_AVAILABLE:
        return None
    try:
        return _GeminiClient()
    except Exception:
        return None


def _get_best_ai_analyst(mozy_client=None):
    \"\"\"
    Smart AI analyst selector:
    - Primary: Mozyfin (if credits available)
    - Fallback: Gemini 2.5 Flash (if Mozyfin exhausted or unavailable)
    Returns: (client, provider_name, credits_info)
    \"\"\"
    # Try Mozyfin first
    if mozy_client:
        try:
            usage = mozy_client.get_usage()
            used = usage.get("credits_used", 0)
            cap = usage.get("credits_cap", 50)
            if used < cap:
                return mozy_client, "mozyfin", usage
        except Exception:
            pass

    # Fallback to Gemini
    gemini = _get_gemini_client()
    if gemini:
        return gemini, "gemini", {}

    return None, "none", {}"""

if old_import in content:
    content = content.replace(old_import, new_import)
    print("✅ AI analyst fallback imports added")
else:
    print("❌ Could not find import block to patch")

# 2. Update the Mozyfin tab to use the smart fallback
old_ai_section = """                # ── AI Analysis (costs 1 credit) ──
                st.markdown("#### \U0001f916 Phân tích AI Mozyfin")
                usage = mozy2.get_usage()
                credits_used = usage.get("credits_used", 0)
                credits_cap = usage.get("credits_cap", 50)
                st.caption(f"Credits đã dùng: {credits_used}/{credits_cap} (1 credit/lần)")
                mozy_key = f"mozyfin_analysis_{clean_sym}"
                if mozy_key not in st.session_state:
                    st.session_state[mozy_key] = ""
                if st.button(
                    f"\U0001f50d Phân tích {clean_sym} bằng Mozyfin AI",
                    key=f"mozy_btn_{clean_sym}",
                    disabled=(credits_used >= credits_cap),
                ):
                    with st.spinner("Mozyfin AI đang phân tích... (~30-60 giây)"):
                        result = mozy2.analyze_stock(clean_sym)
                        st.session_state[mozy_key] = result
                        st.rerun()
                if st.session_state.get(mozy_key):
                    analysis_text = st.session_state[mozy_key]
                    st.markdown("---")
                    st.markdown(analysis_text)
                elif credits_used >= credits_cap:
                    st.warning("Đã hết credits tháng này.")
                else:
                    st.info("Bấm nút để nhận phân tích AI từ dữ liệu báo cáo thực tế.")"""

new_ai_section = """                # ── AI Analysis: Mozyfin primary → Gemini fallback ──
                st.markdown("#### \U0001f916 Phân tích AI")
                analyst, provider, usage = _get_best_ai_analyst(mozy2)

                # Credits display
                if provider == "mozyfin":
                    used = usage.get("credits_used", 0)
                    cap = usage.get("credits_cap", 50)
                    remaining = cap - used
                    st.caption(
                        f"\U0001f7e2 **Mozyfin AI** đang hoạt động | "
                        f"Credits còn lại: **{remaining}/{cap}** | "
                        f"Gemini sẽ thay thế khi hết credit"
                    )
                elif provider == "gemini":
                    st.caption(
                        "\U0001f535 **Gemini 2.5 Flash** (Google AI) — "
                        "Mozyfin hết credit hoặc chưa cấu hình | Miễn phí 1,500 req/ngày"
                    )
                else:
                    st.warning(
                        "Chưa cấu hình AI API. Cần MOZYFIN_API_KEY hoặc GOOGLE_API_KEY "
                        "trong Streamlit Secrets."
                    )

                if analyst:
                    mozy_key = f"ai_analysis_{clean_sym}_{provider}"
                    if mozy_key not in st.session_state:
                        st.session_state[mozy_key] = ""

                    btn_label = (
                        f"\U0001f50d Phân tích {clean_sym} bằng Mozyfin AI"
                        if provider == "mozyfin"
                        else f"\U0001f916 Phân tích {clean_sym} bằng Gemini 2.5 Flash"
                    )
                    if st.button(btn_label, key=f"ai_btn_{clean_sym}_{provider}"):
                        spinner_msg = (
                            "Mozyfin AI đang phân tích... (~30-60 giây)"
                            if provider == "mozyfin"
                            else "Gemini 2.5 Flash đang phân tích... (~10-20 giây)"
                        )
                        with st.spinner(spinner_msg):
                            result = analyst.analyze_stock(clean_sym)
                            st.session_state[mozy_key] = result
                            st.rerun()

                    if st.session_state.get(mozy_key):
                        provider_label = "MOZYFIN AI" if provider == "mozyfin" else "GEMINI 2.5 FLASH"
                        _render_html(
                            f"""<div style="background:rgba(26,26,46,0.95);
                            border:1px solid rgba(102,126,234,0.3);
                            border-radius:12px; padding:20px; line-height:1.8; color:#e2e8f0;">
                            <div style="color:#90cdf4; font-size:0.75rem; letter-spacing:1px;
                            margin-bottom:12px;">{provider_label} ANALYST REPORT — {clean_sym}</div>
                            {st.session_state[mozy_key].replace(chr(10), '<br>').replace('**', '')}
                            </div>"""
                        )
                    else:
                        st.info("Bấm nút để nhận phân tích AI chuyên sâu từ dữ liệu thực tế.")"""

if old_ai_section in content:
    content = content.replace(old_ai_section, new_ai_section)
    print("✅ AI analysis section updated with Gemini fallback")
else:
    print("❌ Could not find AI analysis section to update")
    # Debug: find approximate location
    idx = content.find("credits_used >= credits_cap")
    print(f"  Found 'credits_used >= credits_cap' at char {idx}")

with open(file, "w", encoding="utf-8") as f:
    f.write(content)
print("Done.")
