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
