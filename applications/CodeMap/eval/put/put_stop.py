"""The saturation stop of a GEPA run: stop when K consecutive iterations bring no validation gain above the noise
floor delta. Passed to `gepa.optimize(stop_callbacks=[...])`; gepa 0.1.4 calls it at the top of every iteration
with its state, whose `program_full_scores_val_set` holds the validation score of every accepted candidate and whose
`i` counts iterations (proposals, accepted or rejected).

The anchor is the best validation score seen so far and the iteration it was reached at; it moves only when a
candidate beats it by more than delta, so a gain inside the noise does not reset the streak. The anchor is persisted
beside the run, so `--resume` keeps the streak instead of starting it again. K and delta come from the contract and
the baseline; neither may change after a result is seen (law 1).
"""

import json
import os


class PlateauStopper:
    def __init__(self, k, delta, state_file=None):
        self.k, self.delta, self.state_file = int(k), float(delta), state_file
        self.anchor_score, self.anchor_iteration, self.last_iteration = None, None, None
        self.stopped = False
        if state_file and os.path.exists(state_file):
            with open(state_file, encoding="utf-8") as f:
                saved = json.load(f)
            self.anchor_score, self.anchor_iteration = saved.get("anchor_score"), saved.get("anchor_iteration")

    def _save(self):
        if self.state_file:
            with open(self.state_file, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"k": self.k, "delta": self.delta, "anchor_score": self.anchor_score,
                           "anchor_iteration": self.anchor_iteration, "last_iteration": self.last_iteration,
                           "stopped": self.stopped}, f, indent=1)

    def observe(self, scores, iteration):
        """scores: validation scores of every accepted candidate so far; iteration: GEPA's state.i. -> stop?"""
        self.last_iteration = iteration
        best = max(scores) if scores else None
        if best is not None and (self.anchor_score is None or best > self.anchor_score + self.delta):
            self.anchor_score, self.anchor_iteration = best, iteration
        streak = iteration - (self.anchor_iteration if self.anchor_iteration is not None else iteration)
        self.stopped = streak >= self.k
        self._save()
        return self.stopped

    def streak(self):
        if self.last_iteration is None or self.anchor_iteration is None:
            return 0
        return self.last_iteration - self.anchor_iteration

    def __call__(self, gepa_state):
        return self.observe(list(gepa_state.program_full_scores_val_set), int(gepa_state.i))
