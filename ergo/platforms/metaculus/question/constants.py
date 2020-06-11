# Max loc of 3 set based on API response to prediction on
# https://pandemic.metaculus.com/questions/3920/what-will-the-cbo-estimate-to-be-the-cost-of-the-emergency-telework-act-s3561/
max_loc = 3

# Min loc set based on API response to prediction on
# https://www.metaculus.com/questions/3992/
min_loc = -2

# Max scale of 10 set based on API response to prediction on
# https://pandemic.metaculus.com/questions/3920/what-will-the-cbo-estimate-to-be-the-cost-of-the-emergency-telework-act-s3561/
min_scale = 0.01
max_scale = 10

# We're not really sure what the deal with the low and high is.
# Presumably they're supposed to be the points at which Metaculus "cuts off"
# your distribution and ignores porbability mass assigned below/above.
# But we're not actually trying to use them to "cut off" our distribution
# in a smart way; we're just trying to include as much of our distribution
# as we can without the API getting unhappy
# (we believe that if you set the low higher than the value below
# [or if you set the high lower], then the API will reject the prediction,
# though we haven't tested that extensively)
min_open_low = 0.01
max_open_low = 0.98

# Min high of (low + 0.01) set based on API response for
# https://www.metaculus.com/api2/questions/3961/predict/ --
# {'prediction': ['high minus low must be at least 0.01']}"
min_open_high = 0.01
max_open_high = 0.99
