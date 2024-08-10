def envelope(y, rate, threshold):
  mask = []
  y = pd.Series(y).apply(np.abs)  # make y absolute value 
  y_mean = y.rolling(window = int((rate/10)), min_periods = 1, center = True).mean()
  for mean in y_mean:
    if mean > threshold:
      mask.append(True)
    else:
      mask.append(False)

  return mask


def normalize_mel(S, min_level_db = -100):
    return np.clip((S-min_level_db)/-min_level_db, 0,1)
