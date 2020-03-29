import requests


class MyError(Exception):
    def __init__(self, a, b):
        self.a = a
        self.b = b


message_1 = "this is message 1"
message_2 = "this is message 2"

raise MyError(message_1, message_2)

request = "my request"
response = "my response"

raise requests.HTTPError(
    "the error message", request=request, response=response)

raise requests.
