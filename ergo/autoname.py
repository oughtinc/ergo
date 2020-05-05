from numpyro.primitives import Messenger


class autoname(Messenger):
    """
    If multiple sampling sites have the same name, automatically append a number and
    increment it by 1 for each repeated occurence.
    """

    def __enter__(self):
        self._names = set()
        super(autoname, self).__enter__()

    def _increment_name(self, name, label):
        while (name, label) in self._names:
            try:
                base, count_str = name.rsplit("__", maxsplit=1)
                count = int(count_str) + 1
            except ValueError:
                base, count = name, 1
            name = f"{base}__{count}"
        return name

    def process_message(self, msg):
        if msg["type"] == "sample":
            msg["name"] = self._increment_name(msg["name"], "sample")

    def postprocess_message(self, msg):
        if msg["type"] == "sample":
            self._names.add((msg["name"], "sample"))
