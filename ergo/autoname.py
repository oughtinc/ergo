from numpyro.primitives import Messenger


class autoname(Messenger):
    def __init__(self, fn):
        self._names = set()
        super(autoname, self).__init__(fn)

    def _increment_name(self, name, label):
        while (name, label) in self._names:
            split_name = name.split("__")
            if "__" in name and split_name[-1].isdigit():
                counter = int(split_name[-1]) + 1
                name = "__".join(split_name[:-1] + [str(counter)])
            else:
                name = name + "__1"
        return name

    def process_message(self, msg):
        if msg["type"] == "sample":
            msg["name"] = self._increment_name(msg["name"], "sample")

    def postprocess_message(self, msg):
        if msg["type"] == "sample":
            self._names.add((msg["name"], "sample"))
