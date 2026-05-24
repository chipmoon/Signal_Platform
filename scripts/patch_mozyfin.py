"""Patch ai_forecast.py to add Mozyfin tab."""
import re

file = "views/ai_forecast.py"
with open(file, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add Mozyfin tab to the tabs line
old_tabs = 'tab_chart, tab_vp, tab_ai = st.tabs(["\U0001f680 AI Forecast", " Inst. Flow (VP)", " AI Senate Debate"])'
new_tabs = 'tab_chart, tab_vp, tab_ai, tab_mozy = st.tabs(["\U0001f680 AI Forecast", "\U0001f4ca Inst. Flow (VP)", "\U0001f9e0 AI Senate Debate", "\U0001f1fb\U0001f1f3 Mozyfin Analysis"])'

if old_tabs in content:
    content = content.replace(old_tabs, new_tabs)
    print("✅ Tabs line updated")
else:
    # Try with different whitespace
    pattern = r'tab_chart, tab_vp, tab_ai = st\.tabs\(\[.*?AI Senate Debate.*?\]\)'
    match = re.search(pattern, content)
    if match:
        content = content.replace(match.group(0), new_tabs)
        print(f"✅ Tabs updated via regex: {match.group(0)[:60]}")
    else:
        print("❌ Could not find tabs line!")
        print("Searching for 'tab_chart'...")
        idx = content.find("tab_chart, tab_vp")
        if idx >= 0:
            print(f"Found at char {idx}: {content[idx:idx+100]}")

# 2. Add Mozyfin tab block after the tab_mozy line (before "with tab_chart:")
old_with_chart = "        with tab_chart:"
mozyfin_tab_block = '''        # ── Mozyfin Analysis Tab ─────────────────────────────────────────
        with tab_mozy:
            st.markdown("### \U0001f1fb\U0001f1f3 Mozyfin AI Analysis")
            st.caption("Phân tích bởi Mozyfin AI — dữ liệu từ báo cáo thực tế, tin tức")

            mozy2 = _get_mozyfin_client()
            if not mozy2:
                st.warning("Mozyfin API key chưa cấu hình. Thêm MOZYFIN_API_KEY vào .streamlit/secrets.toml")
            else:
                clean_sym = symbol.replace(".VN", "").replace(".TW", "")
                entity_data = {}
                news_data = []
                try:
                    entity_data = mozy2.search_entity(clean_sym)
                    news_data = mozy2.get_news(clean_sym, limit=5)
                except Exception as _e:
                    logger.debug(f"Mozyfin data: {_e}")

                if entity_data:
                    e1, e2, e3 = st.columns(3)
                    price = entity_data.get("current_price", 0)
                    mcap = entity_data.get("market_cap", 0)
                    e1.metric("Tên", entity_data.get("local_short_name") or entity_data.get("short_name", "-"))
                    e2.metric("Giá hiện tại", f"{price:,.0f}" if price else "-")
                    e3.metric("Vốn hóa", f"{mcap/1e12:.1f}T VND" if mcap else "-")
                    profile = entity_data.get("profile", "")
                    if profile:
                        with st.expander("Giới thiệu doanh nghiệp", expanded=False):
                            st.write(profile[:800] + ("..." if len(profile) > 800 else ""))

                if news_data:
                    st.markdown("#### \U0001f4f0 Tin tức mới nhất")
                    for article in news_data:
                        title = article.get("title", "")
                        content_snip = article.get("content", "")[:200]
                        st.markdown(f"**{title}**")
                        if content_snip:
                            st.caption(content_snip + "...")
                        st.divider()

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
                    st.info("Bấm nút để nhận phân tích AI từ dữ liệu báo cáo thực tế.")

        with tab_chart:'''

if old_with_chart in content:
    content = content.replace(old_with_chart, mozyfin_tab_block, 1)
    print("✅ Mozyfin tab block inserted")
else:
    print("❌ Could not find 'with tab_chart:' to insert Mozyfin tab")

with open(file, "w", encoding="utf-8") as f:
    f.write(content)
print("Done. Run py_compile to verify.")
