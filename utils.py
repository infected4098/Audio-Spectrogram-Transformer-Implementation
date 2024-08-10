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


def summarize_model(model, input_shape, is_cuda = False):
  global device

  if is_cuda:
    x = torch.rand(input_shape).to(device)
  else:
    x = torch.rand(input_shape)
  print(parameter_count_table(model))

  print(summary(model, input_size = input_shape))


def test_model(model, X_test, y_test, device):
  model.eval()
  inputs = X_test
  labels = y_test

  with torch.no_grad():
      inputs = torch.FloatTensor(inputs).to(device)  # (#data, #dim, #timesteps)
      labels = torch.FloatTensor(labels).to(device)  # (#data, #labels)
      utputs = model(inputs)
      preds = torch.argmax(outputs, dim=1)
      labels = torch.argmax(labels)
  return np.array(preds.cpu()), np.array(labels.cpu())

