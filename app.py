"""Scansion AI — 뮤지컬 윤색 자동화 시스템 GUI."""
import os
import sys
import time

import streamlit as st
from dotenv import load_dotenv

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.models import Language, ConstraintSet, LyricCandidate
from core.analyzer import ScansionAnalyzer
from core.generator import LyricGenerator
from core.validator import LyricValidator
from core.music_parser import MusicParser

# .env 파일을 app.py와 같은 디렉토리에서 로드
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_APP_DIR, ".env"))

# Streamlit Cloud secrets 지원: st.secrets에 키가 있으면 환경변수로 설정
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

# ──────────────────────────────────────────
#  페이지 설정
# ──────────────────────────────────────────
st.set_page_config(
    page_title="Scansion AI — 뮤지컬 윤색 자동화",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────
#  커스텀 CSS
# ──────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #888;
        margin-bottom: 2rem;
    }
    .score-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    .score-perfect { border-left: 4px solid #22c55e; }
    .score-good { border-left: 4px solid #eab308; }
    .score-bad { border-left: 4px solid #ef4444; }
    .candidate-card {
        background: #f8f9fa;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.8rem 0;
        transition: all 0.2s;
    }
    .candidate-card:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .syllable-tag {
        display: inline-block;
        background: #e2e8f0;
        padding: 2px 8px;
        border-radius: 4px;
        margin: 1px;
        font-family: monospace;
    }
    .syllable-match { background: #dcfce7; color: #166534; }
    .syllable-mismatch { background: #fee2e2; color: #991b1b; }
    .analysis-box {
        background: #1e1e2e;
        color: #cdd6f4;
        border-radius: 8px;
        padding: 1rem;
        font-family: monospace;
        font-size: 0.85rem;
        white-space: pre-wrap;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1e2e 0%, #2d2d44 100%);
    }
    div[data-testid="stSidebar"] .stMarkdown { color: #cdd6f4; }
    div[data-testid="stSidebar"] label { color: #bac2de !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────
#  세션 상태 초기화
# ──────────────────────────────────────────
def init_session():
    api_key = os.getenv("ANTHROPIC_API_KEY", "")

    if "analyzer" not in st.session_state:
        st.session_state["analyzer"] = ScansionAnalyzer()
    if "generator" not in st.session_state:
        st.session_state["generator"] = LyricGenerator(api_key=api_key)
    if "validator" not in st.session_state:
        st.session_state["validator"] = LyricValidator()
    for key, val in {
        "analysis_result": None,
        "constraint_set": None,
        "candidates": [],
        "selected_candidate": None,
        "generation_count": 0,
        "api_key_set": False,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # 매 실행마다 API 키 상태 확인 및 강제 재설정
    if api_key:
        st.session_state["generator"].set_api_key(api_key)
        st.session_state["api_key_set"] = True

init_session()


# ──────────────────────────────────────────
#  사이드바
# ──────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ 설정")

    # API 키 상태 표시 (키 값은 노출하지 않음)
    st.markdown("### API 연결")
    if st.session_state.api_key_set:
        st.markdown("✅ Claude API 연결됨")
    else:
        # .env에 키가 없을 때만 입력 필드 표시
        api_key_input = st.text_input(
            "Anthropic API Key",
            value="",
            type="password",
            help="Claude API 키를 입력하세요. 또는 .env 파일에 설정하세요.",
        )
        if api_key_input:
            st.session_state.generator.set_api_key(api_key_input)
            st.session_state.api_key_set = True
            st.rerun()
        else:
            st.markdown("❌ API 키 미설정")

    st.divider()

    # 언어 설정
    st.markdown("### 언어 설정")
    source_lang = st.selectbox(
        "원본 언어",
        options=["영어 (English)", "한국어 (Korean)"],
        index=0,
    )
    target_lang = st.selectbox(
        "목표 언어",
        options=["한국어 (Korean)", "영어 (English)"],
        index=0,
    )

    src_lang_enum = Language.EN if "영어" in source_lang else Language.KO
    tgt_lang_enum = Language.KO if "한국어" in target_lang else Language.EN

    st.divider()

    # 생성 설정
    st.markdown("### 생성 설정")
    num_candidates = st.slider("후보 생성 수", 1, 10, 5)
    temperature = st.slider("창의성 (Temperature)", 0.5, 1.0, 0.85, 0.05)
    adaptation_level = st.slider(
        "의역 수준",
        1, 5, 3,
        help="1=직역에 가깝게 / 5=자유롭게 각색",
    )

    st.divider()

    # MusicXML 업로드
    st.markdown("### 악보 업로드 (선택)")
    uploaded_file = st.file_uploader(
        "MusicXML / MIDI 파일",
        type=["xml", "musicxml", "mid", "midi"],
        help="악보를 업로드하면 음표-음절 매핑을 자동 분석합니다.",
    )

    if uploaded_file:
        if MusicParser.is_available():
            try:
                fmt = "musicxml" if uploaded_file.name.endswith(('.xml', '.musicxml')) else "midi"
                parsed = MusicParser.parse_bytes(uploaded_file.read(), fmt)
                st.success(f"파싱 완료: {parsed.title or uploaded_file.name}")
                st.markdown(f"- 박자: {parsed.time_signature}")
                st.markdown(f"- 템포: {parsed.tempo} BPM")
                st.markdown(f"- 음표 수: {len(parsed.notes)}")
            except Exception as e:
                st.error(f"파싱 오류: {e}")
        else:
            st.warning("music21이 설치되지 않았습니다. pip install music21")

    st.divider()
    st.markdown(
        '<div style="text-align:center;color:#666;font-size:0.75rem;">'
        'Scansion AI v1.0<br>뮤지컬 윤색 자동화 시스템'
        '</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────
#  메인 영역
# ──────────────────────────────────────────
st.markdown('<div class="main-header">Scansion AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">뮤지컬 가사 윤색 자동화 시스템 — 음절, 강세, 라임을 지키는 AI 작사</div>', unsafe_allow_html=True)

# 탭 구성
tab_work, tab_analysis, tab_compare, tab_export = st.tabs([
    "🎼 작업", "📊 분석 결과", "🔄 비교", "📤 내보내기"
])

# ══════════════════════════════════════════
#  탭 1: 작업
# ══════════════════════════════════════════
with tab_work:
    col_input, col_output = st.columns([1, 1])

    with col_input:
        st.markdown("### 원본 가사 입력")
        default_text = (
            "Twinkle twinkle little star\n"
            "How I wonder what you are\n"
            "Up above the world so high\n"
            "Like a diamond in the sky"
        )
        source_text = st.text_area(
            "원본 가사 (줄바꿈으로 행 구분)",
            value=default_text,
            height=200,
            key="source_text",
        )

        context = st.text_input(
            "맥락/장면 설명 (선택)",
            placeholder="예: 아이가 밤하늘의 별을 보며 부르는 노래",
            key="context_input",
        )

        character_voice = st.text_input(
            "캐릭터 특성 (선택)",
            placeholder="예: 순수하고 호기심 많은 아이의 목소리",
            key="character_input",
        )

        col_analyze, col_generate = st.columns(2)

        with col_analyze:
            analyze_btn = st.button(
                "🔍 스캔션 분석",
                type="secondary",
                use_container_width=True,
            )

        with col_generate:
            generate_btn = st.button(
                "✨ 윤색 생성",
                type="primary",
                use_container_width=True,
                disabled=not st.session_state.api_key_set,
            )

        if not st.session_state.api_key_set:
            st.info("💡 사이드바에서 Anthropic API 키를 설정하면 윤색 생성이 활성화됩니다.")

    # 분석 실행
    if analyze_btn and source_text.strip():
        with st.spinner("스캔션 분석 중..."):
            analyzer = st.session_state.analyzer
            analysis, constraints = analyzer.analyze_with_constraints(
                source_text,
                source_language=src_lang_enum,
                target_language=tgt_lang_enum,
                context=context,
                adaptation_level=adaptation_level,
            )
            constraints.character_voice = character_voice
            st.session_state.analysis_result = analysis
            st.session_state.constraint_set = constraints

        st.success("분석 완료!")

    # 생성 실행
    if generate_btn and source_text.strip():
        # 먼저 분석이 안 되어 있으면 자동 실행
        if st.session_state.analysis_result is None:
            analyzer = st.session_state.analyzer
            analysis, constraints = analyzer.analyze_with_constraints(
                source_text,
                source_language=src_lang_enum,
                target_language=tgt_lang_enum,
                context=context,
                adaptation_level=adaptation_level,
            )
            constraints.character_voice = character_voice
            st.session_state.analysis_result = analysis
            st.session_state.constraint_set = constraints

        # 생성 전 API 키 재확인
        api_key_now = os.getenv("ANTHROPIC_API_KEY", "")
        if api_key_now:
            st.session_state.generator.set_api_key(api_key_now)
            st.session_state.api_key_set = True

        with st.spinner(f"Claude API로 {num_candidates}개 후보 생성 중... (약 10~20초)"):
            try:
                generator = st.session_state.generator
                validator = st.session_state.validator

                candidates = generator.generate_candidates(
                    st.session_state.constraint_set,
                    num_candidates=num_candidates,
                    temperature=temperature,
                )

                # 검증
                if candidates:
                    candidates = validator.validate_all(
                        candidates,
                        st.session_state.constraint_set,
                        st.session_state.analysis_result,
                    )
                    st.session_state.candidates = candidates
                    st.session_state.generation_count += 1
                    st.success(f"{len(candidates)}개 후보 생성 완료!")
                else:
                    st.error("후보를 생성하지 못했습니다. API 키와 네트워크를 확인하세요.")
            except Exception as e:
                import traceback
                st.error(f"생성 오류: {e}")
                st.code(traceback.format_exc())

    # 우측: 분석 요약 + 후보 표시
    with col_output:
        # 분석 요약
        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            st.markdown("### 분석 요약")

            metric_cols = st.columns(4)
            with metric_cols[0]:
                st.metric("총 행 수", len(result.lines))
            with metric_cols[1]:
                st.metric("총 음절", result.total_syllables)
            with metric_cols[2]:
                st.metric("라임 스킴", result.rhyme_scheme)
            with metric_cols[3]:
                lang_name = "한국어" if result.language == Language.KO else "영어"
                st.metric("언어", lang_name)

            # 행별 음절 분해 시각화
            st.markdown("#### 행별 음절 분석")
            for line in result.lines:
                syl_parts = [s.text for s in line.syllables]
                syl_html = ' '.join(
                    f'<span class="syllable-tag">{s}</span>' for s in syl_parts
                )

                # 강세 표시 (영어)
                stress_str = ""
                if result.language == Language.EN:
                    stress_marks = ['**S**' if s >= 1 else 'w' for s in line.stress_pattern]
                    stress_str = f" | 강세: {' '.join(stress_marks)}"

                rhyme_label = ""
                if result.rhyme_scheme and line.line_number < len(result.rhyme_scheme):
                    rhyme_label = f" [{result.rhyme_scheme[line.line_number]}]"

                st.markdown(
                    f"**행 {line.line_number + 1}** ({line.syllable_count}음절){rhyme_label}: "
                    f"{syl_html}{stress_str}",
                    unsafe_allow_html=True,
                )

        st.divider()

        # 후보 표시
        if st.session_state.candidates:
            st.markdown(f"### 생성된 후보 ({len(st.session_state.candidates)}개)")

            for idx, cand in enumerate(st.session_state.candidates):
                score = cand.scores.total
                if score >= 0.85:
                    badge = "🟢"
                    border_class = "score-perfect"
                elif score >= 0.70:
                    badge = "🟡"
                    border_class = "score-good"
                else:
                    badge = "🔴"
                    border_class = "score-bad"

                with st.expander(
                    f"{badge} 후보 {idx + 1} — 총점: {score:.2f}",
                    expanded=(idx == 0),
                ):
                    # 가사 표시
                    for i, line in enumerate(cand.lines):
                        target_syl = (
                            st.session_state.constraint_set.constraints[i].syllable_count
                            if i < len(st.session_state.constraint_set.constraints)
                            else "?"
                        )
                        actual_syl = cand.syllable_counts[i] if i < len(cand.syllable_counts) else "?"
                        match = "✅" if actual_syl == target_syl else "❌"

                        st.markdown(f"**행 {i+1}**: {line}  ({actual_syl}/{target_syl} 음절 {match})")

                    st.divider()

                    # 점수 세부
                    score_cols = st.columns(6)
                    labels = ["음절", "강세", "라임", "싱어빌리티", "의미", "자연스러움"]
                    values = [
                        cand.scores.syllable_match,
                        cand.scores.stress_match,
                        cand.scores.rhyme_match,
                        cand.scores.singability,
                        cand.scores.semantic,
                        cand.scores.naturalness,
                    ]
                    for col, label, val in zip(score_cols, labels, values):
                        with col:
                            color = "#22c55e" if val >= 0.7 else "#eab308" if val >= 0.4 else "#ef4444"
                            st.markdown(
                                f'<div style="text-align:center;">'
                                f'<div style="font-size:0.75rem;color:#888;">{label}</div>'
                                f'<div style="font-size:1.3rem;font-weight:700;color:{color};">'
                                f'{val:.0%}</div></div>',
                                unsafe_allow_html=True,
                            )

                    # 선택 버튼
                    btn_cols = st.columns([1, 1, 1])
                    with btn_cols[0]:
                        if st.button(f"✅ 선택", key=f"select_{idx}"):
                            st.session_state.selected_candidate = cand
                            st.success("후보가 선택되었습니다!")

                    with btn_cols[1]:
                        if st.button(f"🔄 재생성", key=f"regen_{idx}"):
                            with st.spinner("재생성 중..."):
                                feedback = st.session_state.validator.generate_feedback(
                                    cand, st.session_state.constraint_set,
                                )
                                try:
                                    new_cand = st.session_state.generator.regenerate(
                                        st.session_state.constraint_set,
                                        cand, feedback,
                                    )
                                    if new_cand:
                                        new_cand = st.session_state.validator.validate_candidate(
                                            new_cand, st.session_state.constraint_set,
                                        )
                                        st.session_state.candidates[idx] = new_cand
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"재생성 실패: {e}")


# ══════════════════════════════════════════
#  탭 2: 분석 결과 상세
# ══════════════════════════════════════════
with tab_analysis:
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        constraints = st.session_state.constraint_set

        col_a, col_c = st.columns([1, 1])

        with col_a:
            st.markdown("### 스캔션 분석 결과")
            report = st.session_state.analyzer.format_analysis_report(result)
            st.markdown(f'<div class="analysis-box">{report}</div>', unsafe_allow_html=True)

        with col_c:
            st.markdown("### 윤색 제약조건")
            if constraints:
                report = st.session_state.analyzer.format_constraints_report(constraints)
                st.markdown(f'<div class="analysis-box">{report}</div>', unsafe_allow_html=True)

        st.divider()

        # 음절 패턴 시각화
        st.markdown("### 음절 패턴 시각화")
        import pandas as pd
        chart_df = pd.DataFrame(
            {"음절 수": result.syllable_pattern},
            index=[f"행 {i+1}" for i in range(len(result.syllable_pattern))],
        )
        st.bar_chart(chart_df)

    else:
        st.info("먼저 '🔍 스캔션 분석' 버튼을 클릭하세요.")


# ══════════════════════════════════════════
#  탭 3: 원본 vs 윤색 비교
# ══════════════════════════════════════════
with tab_compare:
    if st.session_state.analysis_result and st.session_state.candidates:
        result = st.session_state.analysis_result
        candidates = st.session_state.candidates

        selected_idx = st.selectbox(
            "비교할 후보 선택",
            options=list(range(len(candidates))),
            format_func=lambda i: f"후보 {i+1} (점수: {candidates[i].scores.total:.2f})",
        )

        cand = candidates[selected_idx]

        st.markdown("### 원본 ↔ 윤색 비교")

        for i, line in enumerate(result.lines):
            if i >= len(cand.lines):
                break

            col_orig, col_arrow, col_adapted = st.columns([5, 1, 5])

            target_syl = line.syllable_count
            actual_syl = cand.syllable_counts[i] if i < len(cand.syllable_counts) else 0
            syl_match = "✅" if actual_syl == target_syl else "❌"

            with col_orig:
                st.markdown(f"**[원본 행 {i+1}]** ({line.syllable_count}음절)")
                st.markdown(f"> {line.original_text}")

                # 음절 분해
                orig_syls = [s.text for s in line.syllables]
                st.markdown("음절: " + " · ".join(orig_syls))

            with col_arrow:
                st.markdown("<div style='text-align:center;font-size:2rem;padding-top:1rem;'>→</div>", unsafe_allow_html=True)

            with col_adapted:
                st.markdown(f"**[윤색 행 {i+1}]** ({actual_syl}음절 {syl_match})")
                st.markdown(f"> {cand.lines[i]}")

                # 음절 분해
                tgt_lang = st.session_state.constraint_set.target_language
                if tgt_lang == Language.KO:
                    from core.korean import KoreanAnalyzer
                    adapted_syls = KoreanAnalyzer.extract_line_syllables(cand.lines[i])
                else:
                    from core.english import EnglishAnalyzer
                    adapted_syls_obj = EnglishAnalyzer.analyze_syllables(cand.lines[i])
                    adapted_syls = [s.text for s in adapted_syls_obj]

                st.markdown("음절: " + " · ".join(adapted_syls))

            st.divider()

        # 검증 리포트
        st.markdown("### 검증 리포트")
        report = st.session_state.validator.format_validation_report(
            cand, st.session_state.constraint_set,
        )
        st.markdown(f'<div class="analysis-box">{report}</div>', unsafe_allow_html=True)

    else:
        st.info("원본 가사를 분석하고 윤색을 생성한 후 비교할 수 있습니다.")


# ══════════════════════════════════════════
#  탭 4: 내보내기
# ══════════════════════════════════════════
with tab_export:
    if st.session_state.candidates:
        st.markdown("### 결과 내보내기")

        selected_for_export = st.session_state.selected_candidate
        if not selected_for_export and st.session_state.candidates:
            selected_for_export = st.session_state.candidates[0]

        if selected_for_export:
            st.markdown("#### 선택된 윤색 가사")
            st.text_area(
                "최종 가사 (편집 가능)",
                value=selected_for_export.text,
                height=200,
                key="export_text",
            )

            # 내보내기 옵션
            st.markdown("#### 내보내기 형식")
            export_cols = st.columns(3)

            with export_cols[0]:
                # 텍스트 다운로드
                export_text = st.session_state.get("export_text", selected_for_export.text)

                # 전체 리포트 텍스트 생성
                full_report = "═══ Scansion AI 윤색 결과 ═══\n\n"

                if st.session_state.analysis_result:
                    src_lang_name = "한국어" if st.session_state.analysis_result.language == Language.KO else "영어"
                    tgt_lang_name = "한국어" if st.session_state.constraint_set.target_language == Language.KO else "영어"
                    full_report += f"방향: {src_lang_name} → {tgt_lang_name}\n"
                    full_report += f"라임 스킴: {st.session_state.analysis_result.rhyme_scheme}\n\n"

                full_report += "── 원본 가사 ──\n"
                if st.session_state.analysis_result:
                    for line in st.session_state.analysis_result.lines:
                        full_report += f"  {line.original_text}\n"

                full_report += "\n── 윤색 가사 ──\n"
                for line in selected_for_export.lines:
                    full_report += f"  {line}\n"

                full_report += "\n── 검증 점수 ──\n"
                full_report += st.session_state.validator.format_validation_report(
                    selected_for_export, st.session_state.constraint_set,
                )

                st.download_button(
                    "📄 텍스트 다운로드",
                    data=full_report.encode("utf-8"),
                    file_name="scansion_result.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            with export_cols[1]:
                # JSON 다운로드
                import json
                export_data = {
                    "source": {
                        "language": st.session_state.analysis_result.language.value if st.session_state.analysis_result else "",
                        "lines": [
                            {
                                "text": l.original_text,
                                "syllable_count": l.syllable_count,
                                "stress_pattern": l.stress_pattern,
                            }
                            for l in st.session_state.analysis_result.lines
                        ] if st.session_state.analysis_result else [],
                        "rhyme_scheme": st.session_state.analysis_result.rhyme_scheme if st.session_state.analysis_result else "",
                    },
                    "adapted": {
                        "lines": [
                            {
                                "text": ln,
                                "syllable_count": selected_for_export.syllable_counts[i] if i < len(selected_for_export.syllable_counts) else 0,
                            }
                            for i, ln in enumerate(selected_for_export.lines)
                        ],
                        "scores": {
                            "total": selected_for_export.scores.total,
                            "syllable_match": selected_for_export.scores.syllable_match,
                            "stress_match": selected_for_export.scores.stress_match,
                            "rhyme_match": selected_for_export.scores.rhyme_match,
                            "singability": selected_for_export.scores.singability,
                            "semantic": selected_for_export.scores.semantic,
                            "naturalness": selected_for_export.scores.naturalness,
                        },
                    },
                }

                st.download_button(
                    "📊 JSON 다운로드",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name="scansion_result.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with export_cols[2]:
                # 대조 텍스트 (양쪽 정렬)
                compare_text = "═══ 원본 ↔ 윤색 대조 ═══\n\n"
                if st.session_state.analysis_result:
                    for i, line in enumerate(st.session_state.analysis_result.lines):
                        adapted_line = selected_for_export.lines[i] if i < len(selected_for_export.lines) else ""
                        orig_syl = line.syllable_count
                        adapted_syl = selected_for_export.syllable_counts[i] if i < len(selected_for_export.syllable_counts) else 0
                        match = "✓" if orig_syl == adapted_syl else "✗"

                        compare_text += f"행 {i+1} [{match}]\n"
                        compare_text += f"  원본 ({orig_syl}음절): {line.original_text}\n"
                        compare_text += f"  윤색 ({adapted_syl}음절): {adapted_line}\n\n"

                st.download_button(
                    "📋 대조표 다운로드",
                    data=compare_text.encode("utf-8"),
                    file_name="scansion_comparison.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
    else:
        st.info("윤색 생성 후 내보내기가 가능합니다.")
