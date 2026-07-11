"""Sweep-line rectangle overlap detection over an interval union tree.

Adapted verbatim from the original module: a segment tree over the
compressed y-coordinate grid tracks covered interval length while a
left-to-right sweep processes rectangle start/end events; regions
where the cover count reaches two or more are recorded together with
the set of participating rectangle indices. Only the entry assertion,
the module docstring, and two shadowing lambda parameter names differ
from the source.
"""


class IntervalUnionQuery:
    def __init__(self, L, y_coords):
        if L == []:
            raise ValueError("interval list must be non-empty")

        self.N = 1
        while self.N < len(L):
            self.N *= 2

        self.c = [0] * (2 * self.N)
        self.s = [0] * (2 * self.N)
        self.w = [0] * (2 * self.N)
        self.overlaps = []
        self.current_x = None
        self.y_coords = y_coords

        self.active_intervals = {}
        self.sweep_events = []
        self.active_rectangles = {}

        for i, _ in enumerate(L):
            self.w[self.N + i] = L[i]
        for p in range(self.N - 1, 0, -1):
            self.w[p] = self.w[2 * p] + self.w[2 * p + 1]

    def union_size(self):
        return self.s[1]

    def modify_interval(self, i, k, offset, x_coord, rect_idx):
        for y in range(i, k):
            if y not in self.active_intervals:
                self.active_intervals[y] = 0
                self.active_rectangles[y] = set()

            old_count = self.active_intervals[y]
            self.active_intervals[y] += offset
            new_count = self.active_intervals[y]

            if offset == 1:
                self.active_rectangles[y].add(rect_idx)
            else:
                self.active_rectangles[y].discard(rect_idx)

            if old_count != new_count:
                self.sweep_events.append(
                    (x_coord, y, old_count, new_count, set(self.active_rectangles[y]))
                )

        self._change(1, 0, self.N, i, k, offset)

    def find_overlaps(self):
        self.sweep_events.sort()
        active_regions = {}

        for x, y, old_count, new_count, rect_set in self.sweep_events:
            if new_count < old_count:
                for count in range(old_count, new_count, -1):
                    if count >= 2 and y in active_regions.get(count, {}):
                        start_x, start_rects = active_regions[count][y]
                        if x - start_x > 0:
                            y_low = self.y_coords[y]
                            y_high = (
                                self.y_coords[y + 1]
                                if y + 1 < len(self.y_coords)
                                else self.y_coords[y]
                            )
                            self.overlaps.append(
                                (start_x, x, y_low, y_high, count, start_rects)
                            )
                        del active_regions[count][y]

            if new_count > old_count:
                for count in range(old_count + 1, new_count + 1):
                    if count >= 2:
                        if count not in active_regions:
                            active_regions[count] = {}
                        active_regions[count][y] = (x, rect_set)

        self.overlaps = [o for o in self.overlaps if o[1] - o[0] > 0]
        self.overlaps.sort(key=lambda o: (-o[4], o[0], o[2]))

    def _change(self, p, start, span, i, k, offset):
        if start + span <= i or k <= start:
            return
        if i <= start and start + span <= k:
            self.c[p] += offset
        else:
            self._change(2 * p, start, span // 2, i, k, offset)
            self._change(2 * p + 1, start + span // 2, span // 2, i, k, offset)

        if self.c[p] == 0:
            if p >= self.N:
                self.s[p] = 0
            else:
                self.s[p] = self.s[2 * p] + self.s[2 * p + 1]
        else:
            self.s[p] = self.w[p]


class Event:
    def __init__(self, x, rectangle, is_start, rect_idx):
        self.x = x
        self.rectangle = rectangle
        self.is_start = is_start
        self.rect_idx = rect_idx


def find_overlapping(rectangles):
    if not rectangles:
        return []

    normalized_rectangles = []
    for rect in rectangles:
        x0, y0, x1, y1 = rect
        if y0 > y1:
            y0, y1 = y1, y0
        normalized_rectangles.append((x0, y0, x1, y1))

    events = []
    for i, rectangle in enumerate(normalized_rectangles):
        events.append(Event(rectangle[0], rectangle, True, i))
        events.append(Event(rectangle[2], rectangle, False, i))

    events.sort(key=lambda e: (e.x, not e.is_start, e.rectangle[1]))

    y_coordinates = sorted(
        list(set([y for rect in normalized_rectangles for y in rect[1::2]]))
    )
    y_intervals = [
        y_coordinates[i + 1] - y_coordinates[i]
        for i in range(len(y_coordinates) - 1)
    ]
    y_mapping = {val: idx for idx, val in enumerate(y_coordinates)}

    interval_query = IntervalUnionQuery(y_intervals, y_coordinates)

    last_x = events[0].x

    for event in events:
        delta_x = event.x - last_x
        last_x = event.x
        rect = event.rectangle
        y0, y1 = y_mapping[rect[1]], y_mapping[rect[3]]

        if event.is_start:
            interval_query.modify_interval(y0, y1, +1, event.x, event.rect_idx)
        else:
            interval_query.modify_interval(y0, y1, -1, event.x, event.rect_idx)

    interval_query.find_overlaps()

    # Return raw overlap index sets for exact edge-to-zone matching
    overlapping_indices = []
    for (start_x, end_x, y_start, y_end, count, rect_indices) in sorted(
        interval_query.overlaps, key=lambda o: (-o[4], o[0])
    ):
        overlapping_indices.append(rect_indices)

    return overlapping_indices