"""Korean language analysis utilities for Scansion AI."""
from __future__ import annotations

from .models import Syllable, StressLevel, RhymeType, RhymeResult

# 초성 (19개)
INITIALS = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

# 중성 (21개)
MEDIALS = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]

# 종성 (28개, 0=없음)
FINALS = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

# 열린 모음 (고음에서 발성하기 좋은 모음)
OPEN_VOWELS = {'ㅏ', 'ㅑ', 'ㅗ', 'ㅛ', 'ㅓ', 'ㅕ', 'ㅣ', 'ㅐ', 'ㅔ'}

# 초성 분류
PLOSIVE_INITIALS = {0, 1, 3, 4, 7, 8, 12, 13, 14, 15, 16, 17}  # ㄱㄲㄷㄸㅂㅃㅈㅉㅊㅋㅌㅍ
NASAL_INITIALS = {2, 6}       # ㄴ ㅁ
LIQUID_INITIALS = {5}         # ㄹ
FRICATIVE_INITIALS = {9, 10, 18}  # ㅅ ㅆ ㅎ
NULL_INITIALS = {11}          # ㅇ

# 유사 모음 그룹 (라임 분석용)
SIMILAR_VOWEL_GROUPS = [
    {0, 2},       # ㅏ, ㅑ
    {4, 6},       # ㅓ, ㅕ
    {8, 11},      # ㅗ, ㅛ
    {12, 17},     # ㅜ, ㅠ
    {1, 5},       # ㅐ, ㅔ
    {3, 7},       # ㅒ, ㅖ
]


class KoreanAnalyzer:
    """한국어 음절/운율 분석기."""

    @staticmethod
    def is_hangul(char: str) -> bool:
        return '가' <= char <= '힣'

    @staticmethod
    def decompose(char: str) -> tuple[int, int, int]:
        """한글 글자를 초성/중성/종성 인덱스로 분해."""
        if not KoreanAnalyzer.is_hangul(char):
            return (-1, -1, -1)
        code = ord(char) - 0xAC00
        initial = code // (21 * 28)
        medial = (code % (21 * 28)) // 28
        final = code % 28
        return (initial, medial, final)

    @staticmethod
    def get_jamo(char: str) -> tuple[str, str, str]:
        """한글 글자를 초성/중성/종성 문자로 분해."""
        i, m, f = KoreanAnalyzer.decompose(char)
        if i == -1:
            return ('', '', '')
        return (INITIALS[i], MEDIALS[m], FINALS[f])

    @staticmethod
    def count_syllables(text: str) -> int:
        """한국어 텍스트의 음절 수 = 한글 글자 수."""
        return sum(1 for c in text if KoreanAnalyzer.is_hangul(c))

    @staticmethod
    def analyze_syllables(text: str) -> list[Syllable]:
        """텍스트를 음절 단위로 분석."""
        syllables = []
        idx = 0
        for char in text:
            if not KoreanAnalyzer.is_hangul(char):
                continue

            initial_idx, medial_idx, final_idx = KoreanAnalyzer.decompose(char)
            has_final = final_idx != 0
            is_open = not has_final

            # 초성 분류
            if initial_idx in PLOSIVE_INITIALS:
                consonant_type = "plosive"
            elif initial_idx in NASAL_INITIALS:
                consonant_type = "nasal"
            elif initial_idx in LIQUID_INITIALS:
                consonant_type = "liquid"
            elif initial_idx in FRICATIVE_INITIALS:
                consonant_type = "fricative"
            else:
                consonant_type = "null"

            # 한국어 의사-강세: 음절 무게 기반
            # 강박에 받침 있는 음절이 오면 자연스러움
            weight = "heavy" if has_final else "light"

            # 모음 기반 음소 표현 (라임 분석용)
            vowel = MEDIALS[medial_idx]
            final_jamo = FINALS[final_idx] if has_final else ""
            phonemes = [vowel]
            if final_jamo:
                phonemes.append(final_jamo)

            syl = Syllable(
                text=char,
                index=idx,
                stress=StressLevel.NONE,  # 한국어는 강세 없음
                phonemes=phonemes,
                is_open=is_open,
                weight=weight,
            )
            syllables.append(syl)
            idx += 1

        return syllables

    @staticmethod
    def get_rhyme_phonemes(char: str) -> list[str]:
        """한국어 각운 분석을 위한 음소 추출 (중성+종성)."""
        if not KoreanAnalyzer.is_hangul(char):
            return []
        _, medial_idx, final_idx = KoreanAnalyzer.decompose(char)
        result = [MEDIALS[medial_idx]]
        if final_idx != 0:
            result.append(FINALS[final_idx])
        return result

    @staticmethod
    def analyze_rhyme(char1: str, char2: str) -> RhymeResult:
        """두 한글 글자의 각운 비교."""
        if not (KoreanAnalyzer.is_hangul(char1) and KoreanAnalyzer.is_hangul(char2)):
            return RhymeResult(type=RhymeType.NONE, score=0.0, word1=char1, word2=char2)

        _, m1, f1 = KoreanAnalyzer.decompose(char1)
        _, m2, f2 = KoreanAnalyzer.decompose(char2)

        # 완전 라임: 중성+종성 일치
        if m1 == m2 and f1 == f2:
            return RhymeResult(type=RhymeType.PERFECT, score=1.0, word1=char1, word2=char2)

        # 모음 라임: 중성만 일치
        if m1 == m2:
            return RhymeResult(type=RhymeType.VOWEL, score=0.7, word1=char1, word2=char2)

        # 자음 라임: 종성만 일치 (둘 다 받침 있을 때)
        if f1 == f2 and f1 != 0:
            return RhymeResult(type=RhymeType.CONSONANCE, score=0.5, word1=char1, word2=char2)

        # 유사 모음
        for group in SIMILAR_VOWEL_GROUPS:
            if m1 in group and m2 in group:
                return RhymeResult(type=RhymeType.NEAR, score=0.4, word1=char1, word2=char2)

        return RhymeResult(type=RhymeType.NONE, score=0.0, word1=char1, word2=char2)

    @staticmethod
    def check_singability(char: str, is_high_note: bool = False, is_fast: bool = False, is_melisma: bool = False) -> float:
        """음절의 싱어빌리티(노래 부르기 적합도) 점수."""
        if not KoreanAnalyzer.is_hangul(char):
            return 0.5

        _, medial_idx, final_idx = KoreanAnalyzer.decompose(char)
        score = 1.0
        vowel = MEDIALS[medial_idx]
        has_final = final_idx != 0

        # 멜리스마 구간: 반드시 열린 음절
        if is_melisma and has_final:
            return 0.0

        # 고음에서는 열린 모음 선호
        if is_high_note:
            if vowel in OPEN_VOWELS and not has_final:
                score = 1.0
            elif vowel in OPEN_VOWELS and has_final:
                score = 0.6
            elif has_final:
                score = 0.4
            else:
                score = 0.7

        # 빠른 패시지에서 받침 있으면 불리
        if is_fast and has_final:
            score -= 0.2

        return max(0.0, min(1.0, score))

    @staticmethod
    def extract_line_syllables(text: str) -> list[str]:
        """텍스트에서 한글 음절만 추출."""
        return [c for c in text if KoreanAnalyzer.is_hangul(c)]
