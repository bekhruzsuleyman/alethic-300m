"""Training items with a loss flag, cut into samples at map events."""

STUB = "<result>\n(dropped; see map)\n</result>"


class Seq:
    def __init__(self):
        self.items = []

    def add(self, text: str, loss: int, kind: str = "") -> int:
        self.items.append({"t": text, "loss": loss, "kind": kind, "map": None})
        return len(self.items) - 1

    def set_read_map(self, read_idx: int, map_idx: int) -> None:
        self.items[read_idx]["map"] = map_idx

    def samples(self, sid: str) -> list:
        """One sample per map event, plus a tail sample if items follow the last map.
        Inside a sample: a read whose map came before the sample's end is stubbed,
        and items already trained in an earlier sample keep loss 0."""
        cuts = [i for i, it in enumerate(self.items) if it["kind"] == "map"]
        last = len(self.items) - 1
        if not cuts or cuts[-1] != last:
            cuts.append(last)
        out, prev = [], -1
        for k, end in enumerate(cuts):
            segs = []
            for i, it in enumerate(self.items[: end + 1]):
                text, loss = it["t"], it["loss"]
                if it["kind"] == "read" and it["map"] is not None and it["map"] < end:
                    text, loss = STUB, 0
                if i <= prev:
                    loss = 0
                segs.append({"t": text, "loss": loss})
            out.append({"id": "%s.%d" % (sid, k), "segments": segs})
            prev = end
        return out
