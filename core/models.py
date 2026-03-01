"""Data models for Scansion AI."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Language(str, Enum):
    EN = "en"
    KO = "ko"


class StressLevel(int, Enum):
    NONE = 0
    PRIMARY = 1
    SECONDARY = 2


class RhymeType(str, Enum):
    PERFECT = "perfect"
    NEAR = "near"
    CONSONANCE = "consonance"
    VOWEL = "vowel"
    NONE = "none"


@dataclass
class Syllable:
    text: str
    index: int
    stress: StressLevel = StressLevel.NONE
    phonemes: list[str] = field(default_factory=list)
    is_open: bool = True  # 열린 음절 (받침 없음)
    weight: str = "light"  # light / heavy


@dataclass
class NoteMapped(Syllable):
    """Syllable mapped to a musical note."""
    note_id: str = ""
    pitch: str = ""
    duration: str = ""
    beat: float = 0.0
    is_downbeat: bool = False
    stress_match: bool = True


@dataclass
class LineAnalysis:
    line_number: int
    original_text: str
    syllables: list[Syllable]
    syllable_count: int
    stress_pattern: list[int] = field(default_factory=list)
    end_phonemes: list[str] = field(default_factory=list)
    rhyme_group: str = ""


@dataclass
class RhymeResult:
    type: RhymeType
    score: float
    word1: str = ""
    word2: str = ""


@dataclass
class ScansionResult:
    language: Language
    lines: list[LineAnalysis]
    rhyme_scheme: str = ""
    syllable_pattern: list[int] = field(default_factory=list)
    total_syllables: int = 0


@dataclass
class Constraint:
    """Constraints for a single line of lyrics."""
    line_number: int
    original_text: str
    original_meaning: str = ""
    syllable_count: int = 0
    stress_pattern: list[int] = field(default_factory=list)
    rhyme_group: str = ""
    rhyme_target_phonemes: list[str] = field(default_factory=list)
    mood: str = ""
    notes: list[dict] = field(default_factory=list)  # note info if MusicXML


@dataclass
class ConstraintSet:
    source_language: Language = Language.EN
    target_language: Language = Language.KO
    constraints: list[Constraint] = field(default_factory=list)
    rhyme_scheme: str = ""
    context: str = ""
    character_voice: str = ""
    adaptation_level: int = 3  # 1=직역 ~ 5=자유각색


@dataclass
class CandidateScore:
    syllable_match: float = 0.0
    stress_match: float = 0.0
    rhyme_match: float = 0.0
    singability: float = 0.0
    semantic: float = 0.0
    naturalness: float = 0.0
    total: float = 0.0

    def calculate_total(
        self,
        weights: Optional[dict] = None,
    ) -> float:
        w = weights or {
            "syllable": 0.30,
            "stress": 0.15,
            "rhyme": 0.15,
            "singability": 0.15,
            "semantic": 0.15,
            "naturalness": 0.10,
        }
        self.total = (
            self.syllable_match * w["syllable"]
            + self.stress_match * w["stress"]
            + self.rhyme_match * w["rhyme"]
            + self.singability * w["singability"]
            + self.semantic * w["semantic"]
            + self.naturalness * w["naturalness"]
        )
        return self.total


@dataclass
class LyricCandidate:
    id: int
    text: str
    lines: list[str]
    syllable_counts: list[int]
    scores: CandidateScore = field(default_factory=CandidateScore)
    generation_round: int = 1
    is_selected: bool = False


@dataclass
class NoteInfo:
    """Parsed note from MusicXML."""
    id: str
    pitch: str
    duration: str
    duration_seconds: float = 0.0
    beat: float = 0.0
    measure: int = 0
    is_downbeat: bool = False
    lyric_syllable: str = ""
    is_melisma: bool = False


@dataclass
class ParsedScore:
    title: str = ""
    time_signature: str = "4/4"
    key: str = "C major"
    tempo: int = 120
    notes: list[NoteInfo] = field(default_factory=list)
    measures: list[list[NoteInfo]] = field(default_factory=list)
