"""English language analysis utilities for Scansion AI."""
from __future__ import annotations

import re
from typing import Optional

from .models import Syllable, StressLevel, RhymeType, RhymeResult

# Try to load pronouncing (CMU dict wrapper)
try:
    import pronouncing
    HAS_PRONOUNCING = True
except ImportError:
    HAS_PRONOUNCING = False

# Try to load pyphen (hyphenation fallback)
try:
    import pyphen
    _PYPHEN_DICT = pyphen.Pyphen(lang='en_US')
    HAS_PYPHEN = True
except ImportError:
    HAS_PYPHEN = False


class EnglishAnalyzer:
    """영어 음절/강세/라임 분석기."""

    @staticmethod
    def count_syllables(text: str) -> int:
        """영어 텍스트의 총 음절 수."""
        words = EnglishAnalyzer._tokenize(text)
        return sum(EnglishAnalyzer._count_word_syllables(w) for w in words)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """텍스트를 단어로 분리."""
        return [w for w in re.findall(r"[a-zA-Z']+", text) if w]

    @staticmethod
    def _count_word_syllables(word: str) -> int:
        """단일 영어 단어의 음절 수."""
        word_lower = word.lower()

        # CMU dict로 정확한 음절 수
        if HAS_PRONOUNCING:
            phones_list = pronouncing.phones_for_word(word_lower)
            if phones_list:
                return pronouncing.syllable_count(phones_list[0])

        # Pyphen 폴백
        if HAS_PYPHEN:
            hyphenated = _PYPHEN_DICT.inserted(word_lower)
            return len(hyphenated.split('-'))

        # 최후의 폴백: 휴리스틱
        return EnglishAnalyzer._heuristic_syllable_count(word_lower)

    @staticmethod
    def _heuristic_syllable_count(word: str) -> int:
        """모음 기반 휴리스틱 음절 수 추정."""
        word = word.lower()
        count = len(re.findall(r'[aeiouy]+', word))
        # 끝의 무음 e
        if word.endswith('e') and count > 1:
            count -= 1
        # -le 패턴
        if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy':
            count += 1
        return max(1, count)

    @staticmethod
    def get_stress_pattern(word: str) -> list[int]:
        """단어의 강세 패턴 반환. [1, 0] = 첫 음절 강세, 둘째 무강세."""
        word_lower = word.lower()

        if HAS_PRONOUNCING:
            phones_list = pronouncing.phones_for_word(word_lower)
            if phones_list:
                stresses = pronouncing.stresses(phones_list[0])
                return [int(s) for s in stresses]

        # 폴백: 영어 기본 패턴 (첫 음절 강세)
        count = EnglishAnalyzer._count_word_syllables(word)
        if count == 1:
            return [1]
        return [1] + [0] * (count - 1)

    @staticmethod
    def analyze_syllables(text: str) -> list[Syllable]:
        """텍스트를 음절 단위로 분석."""
        words = EnglishAnalyzer._tokenize(text)
        syllables: list[Syllable] = []
        idx = 0

        for word in words:
            word_lower = word.lower()
            stress_pattern = EnglishAnalyzer.get_stress_pattern(word)
            expected_count = EnglishAnalyzer._count_word_syllables(word)
            syl_texts = EnglishAnalyzer._syllabify_word(word)

            # syllabify와 CMU 음절 수가 다를 경우 CMU 기준으로 보정
            if len(syl_texts) != expected_count:
                syl_texts = EnglishAnalyzer._force_syllabify(word_lower, expected_count)

            # 음소 정보
            phoneme_groups = EnglishAnalyzer._get_phoneme_groups(word_lower)

            for i, syl_text in enumerate(syl_texts):
                stress_val = stress_pattern[i] if i < len(stress_pattern) else 0
                phonemes = phoneme_groups[i] if i < len(phoneme_groups) else []

                syl = Syllable(
                    text=syl_text,
                    index=idx,
                    stress=StressLevel(min(stress_val, 2)),
                    phonemes=phonemes,
                    is_open=not syl_text[-1].isalpha() or syl_text[-1] in 'aeiouy',
                    weight="heavy" if stress_val >= 1 else "light",
                )
                syllables.append(syl)
                idx += 1

        return syllables

    @staticmethod
    def _force_syllabify(word: str, target_count: int) -> list[str]:
        """CMU dict 음절 수에 맞게 강제 분할."""
        if target_count <= 0:
            return [word]
        if target_count == 1:
            return [word]
        # 균등 분할
        chars = list(word)
        chunk_size = max(1, len(chars) // target_count)
        result = []
        for i in range(target_count):
            start = i * chunk_size
            if i == target_count - 1:
                result.append(''.join(chars[start:]))
            else:
                result.append(''.join(chars[start:start + chunk_size]))
        return [r for r in result if r]

    @staticmethod
    def _syllabify_word(word: str) -> list[str]:
        """단어를 음절로 분리."""
        word_lower = word.lower()

        if HAS_PYPHEN:
            hyphenated = _PYPHEN_DICT.inserted(word_lower)
            parts = hyphenated.split('-')
            if parts:
                return parts

        # 폴백: 모음 기반 분리
        syllables = []
        current = ""
        for ch in word_lower:
            current += ch
            if ch in 'aeiouy' and len(current) > 0:
                syllables.append(current)
                current = ""
        if current:
            if syllables:
                syllables[-1] += current
            else:
                syllables.append(current)
        return syllables if syllables else [word_lower]

    @staticmethod
    def _get_phoneme_groups(word: str) -> list[list[str]]:
        """단어의 음절별 음소 그룹 반환."""
        if not HAS_PRONOUNCING:
            return []

        phones_list = pronouncing.phones_for_word(word.lower())
        if not phones_list:
            return []

        phones = phones_list[0].split()
        groups: list[list[str]] = []
        current: list[str] = []

        for ph in phones:
            current.append(ph)
            # 강세 표시가 있는 음소 = 모음 = 음절 핵
            if ph[-1].isdigit():
                groups.append(current)
                current = []

        # 남은 자음은 마지막 음절에 붙임
        if current and groups:
            groups[-1].extend(current)
        elif current:
            groups.append(current)

        return groups

    @staticmethod
    def get_rhyme_phonemes(word: str) -> list[str]:
        """라임 분석용 음소 추출 (마지막 강세 모음부터 끝까지)."""
        if not HAS_PRONOUNCING:
            # 폴백: 마지막 3글자
            return list(word.lower()[-3:])

        phones_list = pronouncing.phones_for_word(word.lower())
        if not phones_list:
            return list(word.lower()[-3:])

        phones = phones_list[0].split()

        # 뒤에서부터 첫 번째 강세 모음 찾기
        last_stress_idx = -1
        for i in range(len(phones) - 1, -1, -1):
            if phones[i][-1] in '12':
                last_stress_idx = i
                break

        if last_stress_idx >= 0:
            return phones[last_stress_idx:]

        # 강세 없으면 마지막 모음부터
        for i in range(len(phones) - 1, -1, -1):
            if phones[i][-1] == '0':
                return phones[i:]

        return phones[-2:] if len(phones) >= 2 else phones

    @staticmethod
    def analyze_rhyme(word1: str, word2: str) -> RhymeResult:
        """두 영어 단어의 라임 비교."""
        if not HAS_PRONOUNCING:
            # 폴백: 끝 글자 비교
            w1, w2 = word1.lower(), word2.lower()
            if w1[-3:] == w2[-3:]:
                return RhymeResult(type=RhymeType.PERFECT, score=1.0, word1=word1, word2=word2)
            if w1[-2:] == w2[-2:]:
                return RhymeResult(type=RhymeType.NEAR, score=0.6, word1=word1, word2=word2)
            return RhymeResult(type=RhymeType.NONE, score=0.0, word1=word1, word2=word2)

        phon1 = EnglishAnalyzer.get_rhyme_phonemes(word1)
        phon2 = EnglishAnalyzer.get_rhyme_phonemes(word2)

        if phon1 == phon2:
            return RhymeResult(type=RhymeType.PERFECT, score=1.0, word1=word1, word2=word2)

        # 모음만 비교
        vowels1 = [p for p in phon1 if p[0] in 'AEIOU']
        vowels2 = [p for p in phon2 if p[0] in 'AEIOU']
        if vowels1 and vowels1 == vowels2:
            return RhymeResult(type=RhymeType.NEAR, score=0.7, word1=word1, word2=word2)

        # 자음만 비교
        cons1 = [p for p in phon1 if p[0] not in 'AEIOU']
        cons2 = [p for p in phon2 if p[0] not in 'AEIOU']
        if cons1 and cons1 == cons2:
            return RhymeResult(type=RhymeType.CONSONANCE, score=0.5, word1=word1, word2=word2)

        return RhymeResult(type=RhymeType.NONE, score=0.0, word1=word1, word2=word2)

    @staticmethod
    def get_line_stress_pattern(text: str) -> list[int]:
        """한 줄 전체의 강세 패턴."""
        words = EnglishAnalyzer._tokenize(text)
        pattern = []
        for word in words:
            pattern.extend(EnglishAnalyzer.get_stress_pattern(word))
        return pattern
