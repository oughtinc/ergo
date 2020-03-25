import json
import requests
import pendulum
import scipy
import pandas as pd
import numpy as np

from typing import Optional, List


class Metaculus:

  def __init__(self, username, password):
    self.sessionid = None
    self.csrftoken = None
    self.login(username, password)
  
  def login(self, username, password):
    loginURL = "https://www.metaculus.com/api2/accounts/login/"
    r = requests.post(loginURL, 
                      data=json.dumps({"username": username, "password": password}), 
                      headers= { "Content-Type": "application/json" })
    cookie = r.headers['Set-Cookie']
    sessionid = None
    csrftoken = None
    for s in cookie.split(";"):
      s = s.strip()
      if "=" in s:
        k, v = s.split("=")
        if k == "Secure, sessionid":
          sessionid = v
        elif k == "csrftoken":
          csrftoken = v
    
    self.sessionid = sessionid
    self.csrftoken = csrftoken

  def get_question(self, question_id: int, name=None):
    return MetaculusQuestion(question_id, metaculus=self, name=name)


class MetaculusQuestion:
  """
  Attributes:
  - url
  - page_url
  - id
  - author
  - title
  - status
  - resolution
  - created_time
  - publish_time
  - close_time
  - resolve_time
  - possibilities
  - can_use_powers
  - last_activity_time
  - activity
  - comment_count
  - votes
  - prediction_timeseries
  - author_name
  - prediction_histogram
  - anon_prediction_count        
  """
  id: int
  data: Optional[object]
  metaculus: Metaculus
  varname: Optional[str]

  def __init__(self, id: int, metaculus: Metaculus, name=None):
    self.id = id
    self.data = None
    self.metaculus = metaculus
    self.name = name
    self.fetch()
  
  def api_url(self):
    return f"https://www.metaculus.com/api2/questions/{self.id}/"
  
  def fetch(self):
    self.data = requests.get(self.api_url()).json()
    return self.data

  @property
  def is_continuous(self):
    return self.type == "continuous"
  
  @property
  def type(self):
    return self.possibilities["type"]
  
  @property
  def min(self):
    if self.is_continuous:
      return self.possibilities["scale"]["min"]
    return None
  
  @property
  def max(self):
    if self.is_continuous:
      return self.possibilities["scale"]["max"]
    return None
  
  @property
  def deriv_ratio(self):
    if self.is_continuous:
      return self.possibilities["scale"]["deriv_ratio"]
    return None

  @property
  def is_log(self):
    if self.is_continuous:
      return self.deriv_ratio != 1
    return False
    
  def __getattr__(self, name):
    if name in self.data:
      if name.endswith("_time"):
        return pendulum.parse(self.data[name]) # TZ
      return self.data[name]
    else:
      raise AttributeError(name)
  
  def __str__(self):
    if self.data:
      return self.data["title"]
    return "<MetaculusQuestion>"
  
  def submit_from_samples(self, samples, show=False):
    try:
      normalized_samples = self.normalize_samples(samples)
      loc, scale = self.fit_single_logistic(normalized_samples)
    except FloatingPointError:
      print("Error on " + question.area)
      traceback.print_exc()
    else:
      self.submit(loc, scale)
      return (loc, scale)
  
  def submit(self, loc, scale):
    if not self.is_continuous:
      raise NotImplementedError("Can only submit continuous questions!")
    if not self.metaculus:
      raise ValueError("Question was created without Metaculus connection!")

    scale = min(max(scale, 0.02), 10)
    loc = min(max(loc, 0), 1)
    distribution = scipy.stats.logistic(loc, scale)
    
    low = max(distribution.cdf(0), 0.01)
    high = min(distribution.cdf(1), 0.99)
    prediction_data = {
        "prediction":
        {
          "kind":"multi",
          "d":[
              {
                  "kind":"logistic","x0":loc,"s":scale,"w":1,"low":low,"high":high
              }
          ]
        }
        ,
        "void":False
      }
    requests.post(
        f"""https://www.metaculus.com/api2/questions/{self.id}/predict/""", 
        headers={
            "Cookie": f"csrftoken={self.metaculus.csrftoken}; sessionid={self.metaculus.sessionid}", 
            "Content-Type": "application/json",
            "Referer": "https://www.metaculus.com/",
            "X-CSRFToken": self.metaculus.csrftoken
            },
        data=json.dumps(prediction_data)
      )
  
  def normalize_samples(self, samples, epsilon=1e-9):
    if self.is_continuous:
      if self.is_log:
        samples = np.maximum(samples, epsilon)
        samples = samples / self.min
        samples = np.log(samples) / np.log(self.deriv_ratio)
      else:
        samples = (samples - self.min) / (self.max - self.min)
    return samples

  def fit_single_logistic(self, samples):
    with np.errstate(all='raise'):
      loc, scale = scipy.stats.logistic.fit(samples)
      scale = min(max(scale, 0.02), 10)
      loc = min(max(loc, -0.1565), 1.1565)
      return loc, scale

  @classmethod
  def to_dataframe(self, questions: List["MetaculusQuestion"]):
    columns = ["id", "name", "title", "resolve_time"]
    data = []
    for question in questions:
      data.append([question.id, question.name, question.title, question.resolve_time])
    return pd.DataFrame(data, columns=columns)
