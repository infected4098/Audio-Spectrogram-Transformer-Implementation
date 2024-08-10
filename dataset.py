device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X = np.load("/content/gdrive/MyDrive/Audio Classification/models/Xmel_torch.npy", allow_pickle = True) # (#data, #mel_bins, #time bins)
y = np.load("/content/gdrive/MyDrive/Audio Classification/models/ymel_torch.npy") # (#data, )

num_classes = 41

y = np.eye(num_classes)[y] # One-hot-encoder

print(X.shape, y.shape)


class audio_dataset(Dataset):
    def __init__(self, X_data, y_data):

        self.X_data = X_data #(#data, #coefs, #timetsteps)
        self.y_data = y_data #(#data, #labels)
        self.X_torch = torch.FloatTensor(self.X_data).to(device) #(#data, #coefs, #timetsteps)
        self.y_torch = torch.FloatTensor(self.y_data).to(device) #(#data, #labels)
        self.x_len = self.X_torch.shape[0]
        self.y_len = self.y_torch.shape[0]
        assert self.x_len == self.y_len

    def __getitem__(self, index):
        return self.X_torch[index], self.y_torch[index]
    def __len__(self):
        return self.x_len
