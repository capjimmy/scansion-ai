"""LLM-based lyric generation engine using Claude API."""
from __future__ import annotations

import os
import json
import re
import asyncio
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from .models import (
    Language, ConstraintSet, LyricCandidate, CandidateScore, ScansionResult,
)
from .korean import KoreanAnalyzer
from .english import EnglishAnalyzer

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


SYSTEM_PROMPT = """당신은 20년 경력의 뮤지컬 작사가이자 번역가입니다.
영어와 한국어 모두에 능통하며, 다수의 뮤지컬 한국어 윤색 작업에 참여했습니다.

당신의 핵심 역량:
1. 원곡의 멜로디에 맞는 가사를 쓸 수 있습니다
2. 음절 수, 강세, 라임을 정확히 맞출 수 있습니다
3. 단순 번역이 아닌 문화적 각색(윤색)을 할 수 있습니다
4. 캐릭터의 목소리와 감정을 가사에 담을 수 있습니다

반드시 지켜야 할 규칙:
- 음절 수는 반드시 정확히 일치시키세요. 이것은 타협 불가능합니다.
- 한국어 음절 수 = 한글 글자 수입니다. 반드시 정확히 세세요.
- 영어 음절 수는 발음 기준입니다 (예: "twinkle" = 2음절).
- 멜리스마 구간의 음절은 반드시 받침 없는 열린 음절이어야 합니다.
- 고음부(높은 음)에는 가능한 열린 모음(아, 오, 이, 에)을 배치하세요.

출력 규칙:
- 반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 출력하지 마세요.
- JSON 외의 설명, 주석, 마크다운은 절대 포함하지 마세요.
"""

GENERATION_PROMPT_TEMPLATE = """다음 {src_lang} 뮤지컬 가사를 {tgt_lang}로 윤색해주세요.

## 원본 가사
{original_lyrics}

## 제약조건
{constraints_text}

## 전체 라임 스킴
{rhyme_scheme}

## 추가 지시
- 의역 허용 범위: {adaptation_level}/5 (1=직역 ~ 5=자유 각색)
{context_text}
{character_text}

## 출력 형식 (반드시 이 JSON 형식으로만 출력)
{{
  "lines": [
    {{"line": 1, "text": "윤색된 가사", "syllable_count": N, "note": "선택 이유"}},
    ...
  ]
}}
"""

REGENERATION_PROMPT_TEMPLATE = """이전 생성 결과에 대한 피드백입니다. 다시 시도해주세요.

## 원본 가사
{original_lyrics}

## 이전 결과
{previous_result}

## 문제점
{feedback}

## 제약조건 (재확인)
{constraints_text}

## 사용자 피드백
{user_feedback}

위 피드백을 반영하여 다시 생성해주세요.
기존에 문제 없었던 행은 유지하고, 문제가 있는 행만 수정해주세요.

## 출력 형식 (반드시 이 JSON 형식으로만 출력)
{{
  "lines": [
    {{"line": 1, "text": "윤색된 가사", "syllable_count": N, "note": "수정 이유"}},
    ...
  ]
}}
"""


class LyricGenerator:
    """Claude API 기반 가사 생성 엔진."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.model = "claude-sonnet-4-20250514"

    def _get_client(self) -> "anthropic.Anthropic":
        """매 호출마다 클라이언트를 새로 생성 (Streamlit 세션 직렬화 대응)."""
        if not HAS_ANTHROPIC:
            raise RuntimeError("anthropic 패키지가 설치되지 않았습니다.")
        key = self.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        if not key:
            raise RuntimeError(
                "Anthropic API 키가 설정되지 않았습니다. "
                ".env 파일에 ANTHROPIC_API_KEY를 설정하세요."
            )
        return anthropic.Anthropic(api_key=key)

    def is_available(self) -> bool:
        key = self.api_key or os.getenv("ANTHROPIC_API_KEY", "")
        return HAS_ANTHROPIC and bool(key)

    def set_api_key(self, key: str):
        self.api_key = key

    def generate_candidates(
        self,
        constraint_set: ConstraintSet,
        num_candidates: int = 5,
        temperature: float = 0.85,
        user_feedback: str = "",
    ) -> list[LyricCandidate]:
        """제약조건 기반으로 가사 후보 다수 생성."""
        # _get_client()가 키 없으면 RuntimeError를 raise함
        self._get_client()

        prompt = self._build_prompt(constraint_set)
        candidates: list[LyricCandidate] = []

        # 병렬 생성
        with ThreadPoolExecutor(max_workers=min(num_candidates, 5)) as executor:
            futures = []
            for i in range(num_candidates):
                temp = temperature + (i * 0.03)  # 약간의 변화
                temp = min(temp, 1.0)
                futures.append(
                    executor.submit(self._call_llm, prompt, temp, i)
                )

            for future in futures:
                try:
                    candidate = future.result(timeout=60)
                    if candidate:
                        candidates.append(candidate)
                except Exception as e:
                    print(f"생성 실패: {e}")
                    continue

        return candidates

    def regenerate(
        self,
        constraint_set: ConstraintSet,
        previous_candidate: LyricCandidate,
        feedback: str,
        user_feedback: str = "",
        temperature: float = 0.9,
    ) -> Optional[LyricCandidate]:
        """피드백을 반영하여 재생성."""
        if not self.is_available():
            raise RuntimeError("Anthropic API 키가 설정되지 않았습니다.")

        prompt = self._build_regeneration_prompt(
            constraint_set, previous_candidate, feedback, user_feedback,
        )

        return self._call_llm(prompt, temperature, candidate_id=100)

    def _build_prompt(self, cs: ConstraintSet) -> str:
        """제약조건 기반 프롬프트 생성."""
        src_lang = "한국어" if cs.source_language == Language.KO else "영어"
        tgt_lang = "한국어" if cs.target_language == Language.KO else "영어"

        # 원본 가사
        original_lines = []
        for c in cs.constraints:
            original_lines.append(c.original_text)
        original_lyrics = '\n'.join(original_lines)

        # 제약조건 텍스트
        constraint_parts = []
        for c in cs.constraints:
            part = f"행 {c.line_number + 1}: \"{c.original_text}\"\n"
            part += f"  - 필수 음절 수: {c.syllable_count}\n"
            if c.stress_pattern:
                stress_str = ', '.join(str(s) for s in c.stress_pattern)
                part += f"  - 강세 패턴 참고: [{stress_str}]\n"
            if c.rhyme_group:
                part += f"  - 라임 그룹: {c.rhyme_group}\n"
            if c.mood:
                part += f"  - 분위기: {c.mood}\n"
            constraint_parts.append(part)
        constraints_text = '\n'.join(constraint_parts)

        context_text = f"- 맥락: {cs.context}" if cs.context else ""
        character_text = f"- 캐릭터 목소리: {cs.character_voice}" if cs.character_voice else ""

        return GENERATION_PROMPT_TEMPLATE.format(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            original_lyrics=original_lyrics,
            constraints_text=constraints_text,
            rhyme_scheme=cs.rhyme_scheme or "자유",
            adaptation_level=cs.adaptation_level,
            context_text=context_text,
            character_text=character_text,
        )

    def _build_regeneration_prompt(
        self,
        cs: ConstraintSet,
        prev: LyricCandidate,
        feedback: str,
        user_feedback: str,
    ) -> str:
        """재생성 프롬프트 구축."""
        original_lyrics = '\n'.join(c.original_text for c in cs.constraints)

        constraint_parts = []
        for c in cs.constraints:
            part = f"행 {c.line_number + 1}: 필수 음절 수 {c.syllable_count}"
            if c.rhyme_group:
                part += f", 라임 그룹: {c.rhyme_group}"
            constraint_parts.append(part)
        constraints_text = '\n'.join(constraint_parts)

        previous_result = '\n'.join(
            f"행 {i+1}: {line} (음절: {prev.syllable_counts[i] if i < len(prev.syllable_counts) else '?'})"
            for i, line in enumerate(prev.lines)
        )

        return REGENERATION_PROMPT_TEMPLATE.format(
            original_lyrics=original_lyrics,
            previous_result=previous_result,
            feedback=feedback,
            constraints_text=constraints_text,
            user_feedback=user_feedback or "없음",
        )

    def _call_llm(
        self, prompt: str, temperature: float, candidate_id: int,
    ) -> Optional[LyricCandidate]:
        """Claude API 호출."""
        try:
            client = self._get_client()
            response = client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=temperature,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            return self._parse_response(text, candidate_id)

        except Exception as e:
            print(f"LLM 호출 오류: {e}")
            return None

    def _parse_response(self, text: str, candidate_id: int) -> Optional[LyricCandidate]:
        """LLM 응답을 파싱하여 LyricCandidate로 변환."""
        try:
            # JSON 블록 추출 (마크다운 코드블록 처리)
            json_match = re.search(r'\{[\s\S]*\}', text)
            if not json_match:
                return None

            data = json.loads(json_match.group())
            lines_data = data.get("lines", [])

            if not lines_data:
                return None

            lines = [ld["text"] for ld in lines_data]
            full_text = '\n'.join(lines)

            # 실제 음절 수 계산
            syllable_counts = []
            for line in lines:
                if self._is_korean(line):
                    syllable_counts.append(KoreanAnalyzer.count_syllables(line))
                else:
                    syllable_counts.append(EnglishAnalyzer.count_syllables(line))

            return LyricCandidate(
                id=candidate_id,
                text=full_text,
                lines=lines,
                syllable_counts=syllable_counts,
            )

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"응답 파싱 오류: {e}\n원본: {text[:200]}")
            return None

    @staticmethod
    def _is_korean(text: str) -> bool:
        hangul = sum(1 for c in text if '가' <= c <= '힣')
        return hangul > len(text) * 0.3
