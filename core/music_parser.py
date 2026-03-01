"""MusicXML/MIDI parser for Scansion AI using music21."""
from __future__ import annotations

import os
from typing import Optional

from .models import NoteInfo, ParsedScore

try:
    import music21
    HAS_MUSIC21 = True
except ImportError:
    HAS_MUSIC21 = False


class MusicParser:
    """MusicXML / MIDI 파서."""

    @staticmethod
    def is_available() -> bool:
        return HAS_MUSIC21

    @staticmethod
    def parse_file(file_path: str) -> ParsedScore:
        """MusicXML 또는 MIDI 파일을 파싱."""
        if not HAS_MUSIC21:
            raise RuntimeError(
                "music21이 설치되지 않았습니다. "
                "pip install music21 을 실행하세요."
            )

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

        score = music21.converter.parse(file_path)
        return MusicParser._extract_score(score)

    @staticmethod
    def parse_bytes(data: bytes, fmt: str = "musicxml") -> ParsedScore:
        """바이트 데이터를 파싱 (Streamlit 업로드용)."""
        if not HAS_MUSIC21:
            raise RuntimeError("music21이 설치되지 않았습니다.")

        import tempfile
        ext = ".xml" if fmt == "musicxml" else ".mid"
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(data)
            tmp_path = f.name

        try:
            score = music21.converter.parse(tmp_path)
            return MusicParser._extract_score(score)
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _extract_score(score: "music21.stream.Score") -> ParsedScore:
        """music21 Score 객체에서 데이터 추출."""
        # 보컬 파트 찾기
        vocal_part = MusicParser._find_vocal_part(score)

        # 메타데이터
        title = ""
        if score.metadata and score.metadata.title:
            title = score.metadata.title

        time_sig = "4/4"
        for ts in vocal_part.recurse().getElementsByClass('TimeSignature'):
            time_sig = ts.ratioString
            break

        key_sig = "C major"
        for ks in vocal_part.recurse().getElementsByClass('KeySignature'):
            key_sig = str(ks)
            break

        tempo = 120
        for mm in score.recurse().getElementsByClass('MetronomeMark'):
            tempo = int(mm.number)
            break

        # 음표 추출
        all_notes: list[NoteInfo] = []
        measures_grouped: list[list[NoteInfo]] = []

        for measure in vocal_part.getElementsByClass('Measure'):
            measure_notes: list[NoteInfo] = []

            for elem in measure.flat.notesAndRests:
                if isinstance(elem, music21.note.Note):
                    lyric_text = ""
                    if elem.lyric:
                        lyric_text = elem.lyric
                    elif hasattr(elem, 'lyrics') and elem.lyrics:
                        lyric_text = elem.lyrics[0].text if elem.lyrics[0].text else ""

                    # 멜리스마 감지
                    is_melisma = False
                    if hasattr(elem, 'lyrics') and elem.lyrics:
                        for lyr in elem.lyrics:
                            if lyr.syllabic in ('middle', 'end'):
                                is_melisma = True
                                break

                    note_info = NoteInfo(
                        id=f"m{measure.number}_n{len(measure_notes)}",
                        pitch=elem.nameWithOctave,
                        duration=elem.duration.type,
                        duration_seconds=elem.duration.quarterLength * (60.0 / tempo),
                        beat=float(elem.beat),
                        measure=measure.number,
                        is_downbeat=abs(float(elem.beat) - 1.0) < 0.01,
                        lyric_syllable=lyric_text,
                        is_melisma=is_melisma,
                    )
                    all_notes.append(note_info)
                    measure_notes.append(note_info)

            if measure_notes:
                measures_grouped.append(measure_notes)

        return ParsedScore(
            title=title,
            time_signature=time_sig,
            key=key_sig,
            tempo=tempo,
            notes=all_notes,
            measures=measures_grouped,
        )

    @staticmethod
    def _find_vocal_part(score: "music21.stream.Score") -> "music21.stream.Part":
        """보컬 파트 자동 식별."""
        vocal_keywords = ['vocal', 'voice', 'melody', 'singer', 'lead', 'vox']

        # 1. 이름으로 찾기
        for part in score.parts:
            name = (part.partName or "").lower()
            if any(kw in name for kw in vocal_keywords):
                return part

        # 2. 가사가 있는 파트 찾기
        for part in score.parts:
            for note in part.recurse().getElementsByClass('Note'):
                if note.lyric or (hasattr(note, 'lyrics') and note.lyrics):
                    return part

        # 3. 첫 번째 파트
        if score.parts:
            return score.parts[0]

        return score

    @staticmethod
    def create_simple_score(
        syllable_counts: list[int],
        tempo: int = 120,
        time_sig: str = "4/4",
    ) -> ParsedScore:
        """간단한 음표 시퀀스 생성 (MusicXML 없이 텍스트 모드용).

        각 음절에 4분음표 하나씩 배정.
        """
        notes: list[NoteInfo] = []
        measures: list[list[NoteInfo]] = []
        beats_per_measure = int(time_sig.split('/')[0])

        note_idx = 0
        measure_num = 1
        beat_in_measure = 1.0
        current_measure: list[NoteInfo] = []

        for line_idx, count in enumerate(syllable_counts):
            for syl_idx in range(count):
                note = NoteInfo(
                    id=f"m{measure_num}_n{len(current_measure)}",
                    pitch="C4",
                    duration="quarter",
                    duration_seconds=60.0 / tempo,
                    beat=beat_in_measure,
                    measure=measure_num,
                    is_downbeat=abs(beat_in_measure - 1.0) < 0.01,
                )
                notes.append(note)
                current_measure.append(note)
                note_idx += 1

                beat_in_measure += 1.0
                if beat_in_measure > beats_per_measure:
                    measures.append(current_measure)
                    current_measure = []
                    measure_num += 1
                    beat_in_measure = 1.0

        if current_measure:
            measures.append(current_measure)

        return ParsedScore(
            title="Generated",
            time_signature=time_sig,
            tempo=tempo,
            notes=notes,
            measures=measures,
        )
