"""
로그 파일 Tail + ERROR 키워드 감지 모듈
리눅스의 tail -f 동작을 Python으로 구현
여러 로그 파일을 동시에 감시하고 ERROR 발생 시 AI 분석으로 전달
"""
import os
import time
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from config.settings import LOG_FILES, LOG_ERROR_KEYWORDS, LOG_WATCH_INTERVAL_SECONDS


@dataclass
class LogErrorEvent:
    timestamp: str
    log_file: str
    module_name: str       # 파일명에서 추출한 모듈명
    raw_line: str          # 실제 로그 라인
    keyword_matched: str   # 매칭된 키워드 (ERROR, FATAL 등)
    recent_context: list[str]  # 에러 전후 컨텍스트 라인 (최대 5줄)
    context_for_ai: str    # AI 분석용 텍스트


class LogTailer:
    """
    tail -f 방식으로 여러 로그 파일을 동시 감시
    ERROR 키워드 감지 시 콜백 호출
    """

    def __init__(self, base_dir: str = ".", on_error: Callable[[LogErrorEvent], None] = None):
        self.base_dir = base_dir
        self.on_error = on_error
        self._running = False
        self._file_positions: dict[str, int] = {}
        self._line_buffers: dict[str, list[str]] = {}  # 최근 N줄 버퍼
        self._buffer_size = 10
        self._threads: list[threading.Thread] = []

    def _get_module_name(self, filepath: str) -> str:
        return os.path.splitext(os.path.basename(filepath))[0]

    def _tail_file(self, filepath: str):
        abs_path = os.path.join(self.base_dir, filepath)

        # 파일 없으면 생성될 때까지 대기
        while self._running and not os.path.exists(abs_path):
            time.sleep(1)

        # 기존 파일이면 끝으로 이동 (새 로그만 읽기)
        if os.path.exists(abs_path):
            self._file_positions[filepath] = os.path.getsize(abs_path)
        else:
            self._file_positions[filepath] = 0

        self._line_buffers[filepath] = []

        while self._running:
            try:
                current_size = os.path.getsize(abs_path) if os.path.exists(abs_path) else 0

                if current_size > self._file_positions[filepath]:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        f.seek(self._file_positions[filepath])
                        new_lines = f.readlines()
                        self._file_positions[filepath] = f.tell()

                    for line in new_lines:
                        line = line.rstrip()
                        if not line:
                            continue

                        # 라인 버퍼 유지 (컨텍스트용)
                        buf = self._line_buffers[filepath]
                        buf.append(line)
                        if len(buf) > self._buffer_size:
                            buf.pop(0)

                        # ERROR 키워드 감지
                        matched_keyword = self._check_keywords(line)
                        if matched_keyword and self.on_error:
                            event = self._build_event(filepath, line, matched_keyword)
                            self.on_error(event)

            except (FileNotFoundError, PermissionError):
                pass

            time.sleep(LOG_WATCH_INTERVAL_SECONDS)

    def _check_keywords(self, line: str) -> str | None:
        for keyword in LOG_ERROR_KEYWORDS:
            if keyword.upper() in line.upper():
                return keyword
        return None

    def _build_event(self, filepath: str, error_line: str, keyword: str) -> LogErrorEvent:
        module_name = self._get_module_name(filepath)
        context_lines = list(self._line_buffers.get(filepath, []))

        context_for_ai = f"""
[로그 모듈 오류 감지 - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}]

모듈: {module_name}
파일: {filepath}
감지된 키워드: {keyword}

오류 로그:
  {error_line}

최근 로그 컨텍스트 (최대 {self._buffer_size}줄):
{chr(10).join(f"  {line}" for line in context_lines[-5:])}
""".strip()

        return LogErrorEvent(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            log_file=filepath,
            module_name=module_name,
            raw_line=error_line,
            keyword_matched=keyword,
            recent_context=context_lines[-5:],
            context_for_ai=context_for_ai,
        )

    def start(self):
        self._running = True
        for log_file in LOG_FILES:
            t = threading.Thread(
                target=self._tail_file,
                args=(log_file,),
                daemon=True
            )
            t.start()
            self._threads.append(t)
        print(f"[LogTailer] {len(LOG_FILES)}개 파일 감시 시작")

    def stop(self):
        self._running = False
        print("[LogTailer] 중지됨")
