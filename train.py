CONFIG = {
    'lr': 0.0002,
    'epochs': 30,
    'min_batch': 16,
    'weight_decay': 1e-4, # optional, TBD
    "save_path": "/content/gdrive/MyDrive/Audio Classification"
}



X_len = X.shape[0]
train_size = 0.8
idx = np.random.permutation(X_len)
train_idx = idx[:round(train_size*X_len)]; test_idx = idx[round(train_size*X_len):]
X_train = X[train_idx].astype(np.float32); X_test = X[test_idx].astype(np.float32);
y_train = y[train_idx].astype(np.float32); y_test = y[test_idx].astype(np.float32) ;
audio_train = audio_dataset(X_train, y_train)
audio_test = audio_dataset(X_test, y_test)
train_loader = DataLoader(audio_train, batch_size = CONFIG["min_batch"], shuffle = True)
test_loader = DataLoader(audio_test, batch_size = CONFIG["min_batch"], shuffle = True)


model = AST(num_heads = 3, n_blocks = 6).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = CONFIG["lr"])

loss_history = []
val_loss_history = []
init_loss = 991229 # My Birthday

print("Initializing training...")
torch.manual_seed(1229)


for epoch in tqdm(range(CONFIG["epochs"])):
  print(f"\n Epoch {epoch}...")
  run_loss = 0.0

  for i, data in enumerate(train_loader):
    
    inputs, labels = data
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    run_loss += loss.item()


    if i % 500 == 499:
      loss_history.append(run_loss/500)
      print(f'[{epoch +1}, {i + 1:5d}] loss : {run_loss/500:.3f}')
      run_loss = 0.0

      with torch.no_grad():
          val_loss = 0.0
          for k, (val_inputs, val_labels) in enumerate(test_loader):
              val_output = model(val_inputs)
              v_loss = criterion(val_output, val_labels)
              val_loss += v_loss
          print(f'[{epoch + 1}, {i + 1:5d}] val loss: {val_loss / k:.3f}')
          val_loss_history.append(val_loss.item()/500)

      if val_loss < init_loss:
          torch.save(model, os.path.join(CONFIG["save_path"], 'ast.pt'))

          init_loss = val_loss

print("finished training ...")

    
