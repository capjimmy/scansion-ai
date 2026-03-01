"""Validation engine for generated lyrics."""
from __future__ import annotations

from typing import Optional

from .models import (
    Language, ConstraintSet, LyricCandidate, CandidateScore,
    ScansionResult, RhymeType,
)
from .korean import KoreanAnalyzer
from .english import EnglishAnalyzer


class LyricValidator:
    """생성된 가사 후보를 제약조건 대비 검증하고 점수를 매기는 엔진."""

    def __init__(self):
        self.ko = KoreanAnalyzer()
        self.en = EnglishAnalyzer()

    def validate_candidate(
        self,
        candidate: LyricCandidate,
        constraint_set: ConstraintSet,
        source_analysis: Optional[ScansionResult] = None,
    ) -> LyricCandidate:
        """후보 가사를 제약조건 대비 검증 후 점수 계산."""
        lang = constraint_set.target_language
        scores = CandidateScore()

        # 1. 음절 수 매칭 (가장 중요 - hard constraint)
        scores.syllable_match = self._validate_syllable_count(
            candidate, constraint_set,
        )

        # 2. 강세/운율 패턴 매칭
        scores.stress_match = self._validate_stress_pattern(
            candidate, constraint_set, lang,
        )

        # 3. 라임 매칭
        scores.rhyme_match = self._validate_rhyme(
            candidate, constraint_set, lang,
        )

        # 4. 싱어빌리티
        scores.singability = self._validate_singability(
            candidate, lang,
        )

        # 5. 의미 보존 (간이 평가 - LLM 없이)
        scores.semantic = self._validate_semantic_simple(
            candidate, constraint_set,
        )

        # 6. 자연스러움 (간이 평가)
        scores.naturalness = self._validate_naturalness(
            candidate, lang,
        )

        # 가중치 적용하여 총점 계산
        # 한국어 타겟일 때는 강세 가중치를 낮춤
        if lang == Language.KO:
            weights = {
                "syllable": 0.35,
                "stress": 0.05,
                "rhyme": 0.15,
                "singability": 0.15,
                "semantic": 0.20,
                "naturalness": 0.10,
            }
        else:
            weights = {
                "syllable": 0.30,
                "stress": 0.15,
                "rhyme": 0.15,
                "singability": 0.15,
                "semantic": 0.15,
                "naturalness": 0.10,
            }

        scores.calculate_total(weights)
        candidate.scores = scores
        return candidate

    def validate_all(
        self,
        candidates: list[LyricCandidate],
        constraint_set: ConstraintSet,
        source_analysis: Optional[ScansionResult] = None,
    ) -> list[LyricCandidate]:
        """모든 후보를 검증하고 점수순으로 정렬."""
        validated = []
        for candidate in candidates:
            validated_candidate = self.validate_candidate(
                candidate, constraint_set, source_analysis,
            )
            validated.append(validated_candidate)

        # 점수 내림차순 정렬
        validated.sort(key=lambda c: c.scores.total, reverse=True)

        return validated

    def _validate_syllable_count(
        self,
        candidate: LyricCandidate,
        constraint_set: ConstraintSet,
    ) -> float:
        """음절 수 검증. 정확히 일치하면 1.0, 불일치 시 감점."""
        lang = constraint_set.target_language
        constraints = constraint_set.constraints

        if not candidate.lines:
            return 0.0

        total_score = 0.0
        num_lines = min(len(candidate.lines), len(constraints))

        for i in range(num_lines):
            target_count = constraints[i].syllable_count

            if lang == Language.KO:
                actual_count = KoreanAnalyzer.count_syllables(candidate.lines[i])
            else:
                actual_count = EnglishAnalyzer.count_syllables(candidate.lines[i])

            # 업데이트 실제 음절 수
            if i < len(candidate.syllable_counts):
                candidate.syllable_counts[i] = actual_count
            else:
                candidate.syllable_counts.append(actual_count)

            if actual_count == target_count:
                total_score += 1.0
            else:
                diff = abs(actual_count - target_count)
                # 1개 차이: 0.5, 2개 차이: 0.2, 3개 이상: 0
                if diff == 1:
                    total_score += 0.5
                elif diff == 2:
                    total_score += 0.2
                else:
                    total_score += 0.0

        return total_score / max(num_lines, 1)

    def _validate_stress_pattern(
        self,
        candidate: LyricCandidate,
        constraint_set: ConstraintSet,
        lang: Language,
    ) -> float:
        """강세/운율 패턴 검증."""
        if lang == Language.KO:
            # 한국어는 강세가 없으므로 음절 무게(받침 유무) 기반
            return self._validate_korean_weight_pattern(candidate, constraint_set)
        else:
            return self._validate_english_stress(candidate, constraint_set)

    def _validate_english_stress(
        self,
        candidate: LyricCandidate,
        constraint_set: ConstraintSet,
    ) -> float:
        """영어 강세 패턴 검증."""
        total = 0.0
        count = 0

        for i, line in enumerate(candidate.lines):
            if i >= len(constraint_set.constraints):
                break

            target_pattern = constraint_set.constraints[i].stress_pattern
            if not target_pattern:
                continue

            actual_pattern = EnglishAnalyzer.get_line_stress_pattern(line)

            # 두 패턴의 길이가 다를 수 있으므로 짧은 쪽 기준
            min_len = min(len(target_pattern), len(actual_pattern))
            if min_len == 0:
                continue

            matches = sum(
                1 for j in range(min_len)
                if (target_pattern[j] >= 1) == (actual_pattern[j] >= 1)
            )
            total += matches / min_len
            count += 1

        return total / max(count, 1)

    def _validate_korean_weight_pattern(
        self,
        candidate: LyricCandidate,
        constraint_set: ConstraintSet,
    ) -> float:
        """한국어 음절 무게 패턴 검증 (강세 대체)."""
        # 한국어에서는 이 검증이 덜 중요하므로 기본 0.7 부여
        total = 0.0
        count = 0

        for line in candidate.lines:
            syllables = KoreanAnalyzer.analyze_syllables(line)
            if not syllables:
                continue

            # 첫 음절과 마지막 음절이 자연스러운지 체크
            score = 0.7  # 기본 점수

            # 첫 음절이 파열음/비음으로 시작하면 좋음
            if syllables and syllables[0].phonemes:
                score += 0.1

            # 마지막 음절이 열린 음절이면 여운이 좋음
            if syllables and syllables[-1].is_open:
                score += 0.1

            total += min(score, 1.0)
            count += 1

        return total / max(count, 1)

    def _validate_rhyme(
        self,
        candidate: LyricCandidate,
        constraint_set: ConstraintSet,
        lang: Language,
    ) -> float:
        """라임 검증."""
        rhyme_scheme = constraint_set.rhyme_scheme
        if not rhyme_scheme or len(candidate.lines) < 2:
            return 0.7  # 라임 스킴이 없으면 중간 점수

        # 같은 라임 그룹끼리 비교
        groups: dict[str, list[str]] = {}
        for i, line in enumerate(candidate.lines):
            if i >= len(rhyme_scheme):
                break
            group = rhyme_scheme[i]
            # 마지막 단어/음절 추출
            if lang == Language.KO:
                chars = KoreanAnalyzer.extract_line_syllables(line)
                end = chars[-1] if chars else ""
            else:
                import re
                words = re.findall(r"[a-zA-Z']+", line)
                end = words[-1] if words else ""

            if group not in groups:
                groups[group] = []
            groups[group].append(end)

        # 같은 그룹 내 라임 점수
        total_score = 0.0
        total_pairs = 0

        for group, ends in groups.items():
            if len(ends) < 2:
                continue

            for i in range(len(ends)):
                for j in range(i + 1, len(ends)):
                    if lang == Language.KO:
                        result = KoreanAnalyzer.analyze_rhyme(ends[i], ends[j])
                    else:
                        result = EnglishAnalyzer.analyze_rhyme(ends[i], ends[j])
                    total_score += result.score
                    total_pairs += 1

        if total_pairs == 0:
            return 0.7

        return total_score / total_pairs

    def _validate_singability(
        self,
        candidate: LyricCandidate,
        lang: Language,
    ) -> float:
        """싱어빌리티 검증."""
        if lang == Language.KO:
            return self._validate_korean_singability(candidate)
        else:
            return self._validate_english_singability(candidate)

    def _validate_korean_singability(self, candidate: LyricCandidate) -> float:
        """한국어 싱어빌리티."""
        total = 0.0
        count = 0

        for line in candidate.lines:
            syllables = KoreanAnalyzer.analyze_syllables(line)
            if not syllables:
                continue

            line_score = 0.0
            for syl in syllables:
                # 기본 싱어빌리티 체크
                s = KoreanAnalyzer.check_singability(syl.text)
                line_score += s

            total += line_score / max(len(syllables), 1)
            count += 1

        return total / max(count, 1)

    def _validate_english_singability(self, candidate: LyricCandidate) -> float:
        """영어 싱어빌리티 (간이)."""
        total = 0.0
        count = 0

        for line in candidate.lines:
            syllables = EnglishAnalyzer.analyze_syllables(line)
            if not syllables:
                continue

            score = 0.8  # 영어는 기본적으로 괜찮다고 가정

            # 너무 긴 단어는 싱어빌리티 떨어짐
            import re
            words = re.findall(r"[a-zA-Z']+", line)
            long_words = [w for w in words if len(w) > 10]
            if long_words:
                score -= 0.1 * len(long_words)

            total += max(0.0, score)
            count += 1

        return total / max(count, 1)

    def _validate_semantic_simple(
        self,
        candidate: LyricCandidate,
        constraint_set: ConstraintSet,
    ) -> float:
        """의미 보존 간이 검증 (LLM 없이).

        키워드 기반으로 원문의 핵심 단어가 번역에 반영되었는지 체크.
        정확한 평가는 LLM 기반 검증이 필요하지만, 기본 점수를 매김.
        """
        # 간이 검증: 빈 줄이 없는지, 길이가 적절한지
        if not candidate.lines:
            return 0.0

        empty_lines = sum(1 for ln in candidate.lines if not ln.strip())
        if empty_lines > 0:
            return 0.3

        # 모든 줄에 내용이 있으면 기본 0.7
        return 0.7

    def _validate_naturalness(
        self,
        candidate: LyricCandidate,
        lang: Language,
    ) -> float:
        """자연스러움 간이 검증."""
        if not candidate.lines:
            return 0.0

        score = 0.8  # 기본

        for line in candidate.lines:
            # 같은 글자 3번 이상 연속 반복 체크
            if any(c * 3 in line for c in set(line) if c.strip()):
                score -= 0.1

            # 너무 짧거나 긴 줄
            if len(line.strip()) < 2:
                score -= 0.2

        return max(0.0, min(1.0, score))

    def generate_feedback(
        self,
        candidate: LyricCandidate,
        constraint_set: ConstraintSet,
    ) -> str:
        """검증 결과를 기반으로 피드백 텍스트 생성."""
        lang = constraint_set.target_language
        feedbacks = []

        for i, line in enumerate(candidate.lines):
            if i >= len(constraint_set.constraints):
                break

            target_count = constraint_set.constraints[i].syllable_count

            if lang == Language.KO:
                actual_count = KoreanAnalyzer.count_syllables(line)
            else:
                actual_count = EnglishAnalyzer.count_syllables(line)

            if actual_count != target_count:
                diff = actual_count - target_count
                if diff > 0:
                    feedbacks.append(
                        f"행 {i+1}: 음절 수 {actual_count}개 "
                        f"(필요: {target_count}개) → {diff}음절 줄여야 함"
                    )
                else:
                    feedbacks.append(
                        f"행 {i+1}: 음절 수 {actual_count}개 "
                        f"(필요: {target_count}개) → {-diff}음절 늘려야 함"
                    )

        if not feedbacks:
            feedbacks.append("모든 제약조건을 충족합니다.")

        return '\n'.join(feedbacks)

    def format_validation_report(
        self,
        candidate: LyricCandidate,
        constraint_set: ConstraintSet,
    ) -> str:
        """검증 결과 리포트 생성."""
        s = candidate.scores
        lang = constraint_set.target_language

        lines = []
        lines.append("═══ 검증 결과 ═══")
        lines.append(f"총점: {s.total:.2f} / 1.00")
        lines.append("")
        lines.append(f"  음절 매칭: {s.syllable_match:.2f}  {'✓' if s.syllable_match >= 0.9 else '△' if s.syllable_match >= 0.5 else '✗'}")
        lines.append(f"  강세 매칭: {s.stress_match:.2f}  {'✓' if s.stress_match >= 0.7 else '△' if s.stress_match >= 0.4 else '✗'}")
        lines.append(f"  라임 매칭: {s.rhyme_match:.2f}  {'✓' if s.rhyme_match >= 0.7 else '△' if s.rhyme_match >= 0.4 else '✗'}")
        lines.append(f"  싱어빌리티: {s.singability:.2f}  {'✓' if s.singability >= 0.7 else '△' if s.singability >= 0.4 else '✗'}")
        lines.append(f"  의미 보존: {s.semantic:.2f}  {'✓' if s.semantic >= 0.7 else '△' if s.semantic >= 0.4 else '✗'}")
        lines.append(f"  자연스러움: {s.naturalness:.2f}  {'✓' if s.naturalness >= 0.7 else '△' if s.naturalness >= 0.4 else '✗'}")
        lines.append("")

        # 행별 상세
        for i, line in enumerate(candidate.lines):
            if i >= len(constraint_set.constraints):
                break

            target = constraint_set.constraints[i].syllable_count
            actual = candidate.syllable_counts[i] if i < len(candidate.syllable_counts) else '?'

            match_mark = '✓' if actual == target else '✗'
            lines.append(f"  행 {i+1}: \"{line}\"")
            lines.append(f"    음절: {actual}/{target} {match_mark}")

        return '\n'.join(lines)
