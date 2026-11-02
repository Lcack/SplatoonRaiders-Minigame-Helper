"""Per-frame ending detection; full menu scan before play and once per second."""
import time

from raiders_helper.session_runtime import ScreenMatcher


class RuntimeUI:
    def __init__(self,config,root):
        config={**config,'stable_frames':1}
        self.full=ScreenMatcher(config,root)
        self.ending=ScreenMatcher({**config,'screens':[s for s in config['screens']
            if s['name'].startswith(('result:','pause:'))]},root)
        self.last_full=None;self.scope='full_ui_scan';self.ms=0.

    def observe(self,frame,in_round,now):
        full=not in_round or self.last_full is None or now-self.last_full>=1
        if full:self.last_full=now
        self.scope='full_ui_scan' if full else 'runtime_result_only'
        started=time.perf_counter();result=(self.full if full else self.ending).observe(frame)
        self.ms=(time.perf_counter()-started)*1000
        return result
