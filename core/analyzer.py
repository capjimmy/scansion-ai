"""Core scansion analysis engine."""
from __future__ import annotations

import re
from typing import Optional

from .models import (
    Language, LineAnalysis, ScansionResult, Constraint, ConstraintSet,
    ParsedScore, StressLevel,
)
from .korean import KoreanAnalyzer
from .english import EnglishAnalyzer


class ScansionAnalyzer:
    """스캔션 분석 통합 엔진.

    텍스트를 분석하여 음절 수, 강세 패턴, 라임 스킴을 추출하고,
    윤색에 필요한 제약조건을 생성한다.
    """

    def __init__(self):
        self.ko = KoreanAnalyzer()
        self.en = EnglishAnalyzer()

    def detect_language(self, text: str) -> Language:
        """텍스트의 언어 자동 감지."""
        hangul_count = sum(1 for c in text if '가' <= c <= '힣')
        ascii_count = sum(1 for c in text if c.isascii() and c.isalpha())

        if hangul_count > ascii_count:
            return Language.KO
        return Language.EN

    def analyze(
        self,
        text: str,
        language: Optional[Language] = None,
        parsed_score: Optional[ParsedScore] = None,
    ) -> ScansionResult:
        """텍스트의 스캔션 분석 실행."""
        if language is None:
            language = self.detect_language(text)

        lines_text = [ln.strip() for ln in text.strip().split('\n') if ln.strip()]
        lines: list[LineAnalysis] = []

        for i, line_text in enumerate(lines_text):
            if language == Language.KO:
                line_analysis = self._analyze_korean_line(i, line_text)
            else:
                line_analysis = self._analyze_english_line(i, line_text)
            lines.append(line_analysis)

        # 라임 스킴 감지
        rhyme_scheme = self._detect_rhyme_scheme(lines, language)

        # 음절 패턴
        syllable_pattern = [ln.syllable_count for ln in lines]

        return ScansionResult(
            language=language,
            lines=lines,
            rhyme_scheme=rhyme_scheme,
            syllable_pattern=syllable_pattern,
            total_syllables=sum(syllable_pattern),
        )

    def _analyze_korean_line(self, line_num: int, text: str) -> LineAnalysis:
        """한국어 한 줄 분석."""
        syllables = self.ko.analyze_syllables(text)
        syllable_count = len(syllables)

        # 한국어는 강세 대신 음절 무게 패턴
        stress_pattern = [
            1 if s.weight == "heavy" else 0 for s in syllables
        ]

        # 각운 음소 (마지막 음절)
        hangul_chars = self.ko.extract_line_syllables(text)
        end_phonemes = []
        if hangul_chars:
            end_phonemes = self.ko.get_rhyme_phonemes(hangul_chars[-1])

        return LineAnalysis(
            line_number=line_num,
            original_text=text,
            syllables=syllables,
            syllable_count=syllable_count,
            stress_pattern=stress_pattern,
            end_phonemes=end_phonemes,
        )

    def _analyze_english_line(self, line_num: int, text: str) -> LineAnalysis:
        """영어 한 줄 분석."""
        syllables = self.en.analyze_syllables(text)
        syllable_count = len(syllables)
        stress_pattern = self.en.get_line_stress_pattern(text)

        # 각운 음소 (마지막 단어)
        words = re.findall(r"[a-zA-Z']+", text)
        end_phonemes = []
        if words:
            end_phonemes = self.en.get_rhyme_phonemes(words[-1])

        return LineAnalysis(
            line_number=line_num,
            original_text=text,
            syllables=syllables,
            syllable_count=syllable_count,
            stress_pattern=stress_pattern,
            end_phonemes=end_phonemes,
        )

    def _detect_rhyme_scheme(
        self, lines: list[LineAnalysis], language: Language,
    ) -> str:
        """라임 스킴 자동 감지."""
        if len(lines) < 2:
            return "A"

        scheme: list[str] = []
        groups: dict[str, list[list[str]]] = {}
        current_label = 'A'

        for line in lines:
            matched_label = None

            for label, group_phonemes in groups.items():
                for gp in group_phonemes:
                    score = self._compare_rhyme_phonemes(
                        line.end_phonemes, gp, language
                    )
                    if score >= 0.6:
                        matched_label = label
                        break
                if matched_label:
                    break

            if matched_label:
                scheme.append(matched_label)
                groups[matched_label].append(line.end_phonemes)
            else:
                scheme.append(current_label)
                groups[current_label] = [line.end_phonemes]
                current_label = chr(ord(current_label) + 1)
                if current_label > 'Z':
                    current_label = 'A'

        return ''.join(scheme)

    def _compare_rhyme_phonemes(
        self, phon1: list[str], phon2: list[str], language: Language,
    ) -> float:
        """두 음소 세트의 라임 유사도 비교."""
        if not phon1 or not phon2:
            return 0.0

        if phon1 == phon2:
            return 1.0

        if language == Language.KO:
            # 한국어: 첫 번째 원소가 모음
            if len(phon1) > 0 and len(phon2) > 0 and phon1[0] == phon2[0]:
                return 0.7  # 모음 일치
            return 0.0
        else:
            # 영어: 음소 리스트 부분 비교
            vowels1 = [p for p in phon1 if len(p) > 0 and p[0] in 'AEIOU']
            vowels2 = [p for p in phon2 if len(p) > 0 and p[0] in 'AEIOU']
            if vowels1 and vowels1 == vowels2:
                return 0.7
            return 0.0

    def generate_constraints(
        self,
        source_analysis: ScansionResult,
        target_language: Language,
        context: str = "",
        adaptation_level: int = 3,
        character_voice: str = "",
    ) -> ConstraintSet:
        """분석 결과를 기반으로 윤색 제약조건 생성."""
        constraints: list[Constraint] = []

        for line in source_analysis.lines:
            constraint = Constraint(
                line_number=line.line_number,
                original_text=line.original_text,
                syllable_count=line.syllable_count,
                stress_pattern=line.stress_pattern,
                rhyme_group=line.rhyme_group if line.rhyme_group else "",
                rhyme_target_phonemes=line.end_phonemes,
            )

            # 라임 그룹 할당
            if source_analysis.rhyme_scheme and line.line_number < len(source_analysis.rhyme_scheme):
                constraint.rhyme_group = source_analysis.rhyme_scheme[line.line_number]

            constraints.append(constraint)

        return ConstraintSet(
            source_language=source_analysis.language,
            target_language=target_language,
            constraints=constraints,
            rhyme_scheme=source_analysis.rhyme_scheme,
            context=context,
            character_voice=character_voice,
            adaptation_level=adaptation_level,
        )

    def analyze_with_constraints(
        self,
        source_text: str,
        source_language: Optional[Language] = None,
        target_language: Optional[Language] = None,
        context: str = "",
        adaptation_level: int = 3,
    ) -> tuple[ScansionResult, ConstraintSet]:
        """분석 + 제약조건 생성을 한번에 수행."""
        if source_language is None:
            source_language = self.detect_language(source_text)

        if target_language is None:
            target_language = Language.KO if source_language == Language.EN else Language.EN

        analysis = self.analyze(source_text, source_language)
        constraints = self.generate_constraints(
            analysis, target_language, context, adaptation_level,
        )

        return analysis, constraints

    def format_analysis_report(self, result: ScansionResult) -> str:
        """분석 결과를 읽기 좋은 텍스트 리포트로 변환."""
        report_lines = []
        report_lines.append(f"═══ 스캔션 분석 결과 ═══")
        report_lines.append(f"언어: {'한국어' if result.language == Language.KO else '영어'}")
        report_lines.append(f"총 행 수: {len(result.lines)}")
        report_lines.append(f"총 음절 수: {result.total_syllables}")
        report_lines.append(f"라임 스킴: {result.rhyme_scheme}")
        report_lines.append(f"음절 패턴: {result.syllable_pattern}")
        report_lines.append("")

        for line in result.lines:
            report_lines.append(f"── 행 {line.line_number + 1} ──")
            report_lines.append(f"  텍스트: {line.original_text}")
            report_lines.append(f"  음절 수: {line.syllable_count}")

            # 음절 분해
            syl_texts = [s.text for s in line.syllables]
            report_lines.append(f"  음절 분해: {' - '.join(syl_texts)}")

            # 강세 패턴
            if result.language == Language.EN:
                stress_str = ''.join(
                    'S' if s >= 1 else 'w' for s in line.stress_pattern
                )
                report_lines.append(f"  강세 패턴: {stress_str}  (S=강세, w=약세)")

            # 각운 음소
            if line.end_phonemes:
                report_lines.append(f"  각운 음소: {line.end_phonemes}")

            report_lines.append("")

        return '\n'.join(report_lines)

    def format_constraints_report(self, cs: ConstraintSet) -> str:
        """제약조건을 읽기 좋은 리포트로 변환."""
        src_name = '한국어' if cs.source_language == Language.KO else '영어'
        tgt_name = '한국어' if cs.target_language == Language.KO else '영어'

        lines = []
        lines.append(f"═══ 윤색 제약조건 ═══")
        lines.append(f"방향: {src_name} → {tgt_name}")
        lines.append(f"라임 스킴: {cs.rhyme_scheme}")
        lines.append(f"의역 수준: {cs.adaptation_level}/5")
        lines.append("")

        for c in cs.constraints:
            lines.append(f"── 행 {c.line_number + 1} ──")
            lines.append(f"  원문: {c.original_text}")
            lines.append(f"  필요 음절 수: {c.syllable_count}")
            if c.stress_pattern:
                lines.append(f"  강세 패턴: {c.stress_pattern}")
            if c.rhyme_group:
                lines.append(f"  라임 그룹: {c.rhyme_group}")
            lines.append("")

        return '\n'.join(lines)
