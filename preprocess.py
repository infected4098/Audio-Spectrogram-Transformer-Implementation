from utils import envelope, normalize_mel

m_path = "/content/gdrive/MyDrive/Audio Classification/audio_train" # 수정
df = pd.read_csv("/content/gdrive/MyDrive/Audio Classification/real_df.csv") # 수정
df.set_index("fname", inplace = True)

if len(os.listdir("/content/gdrive/MyDrive/Audio Classification/clean")) == 0:
  for f in tqdm(df.fname):
    signal, rate = librosa.load(os.path.join(m_path, f), sr = 16000)
    mask = envelope(signal, rate, 0.0005)
    wavfile.write(filename = "/content/gdrive/MyDrive/Audio Classification/clean/"+f, rate = rate, data = signal[mask])

classes = df["label"].unique()
class_dist = df.groupby(["label"])["length"].mean() #Mean duration for every class ex: saxophone = 3.2 ...
n_samples = 3 * int(df["length"].sum()/10)  # how many times in sampling.
prob_dist = class_dist/class_dist.sum() #sampling probability proportional to n_samples
step = int(128*99) # sampling for about 0.7 seconds.

def build_rand_feat():
  random.seed(1229)
  X = []
  labels = []
  _min, _max = float("inf"), -float("inf")
  for _ in tqdm(range(n_samples)):
    rand_class = np.random.choice(class_dist.index, p = prob_dist)
    file = np.random.choice(df[df.label == rand_class].index)  # RandomSample a file in the class category
    try:

      wav, sr = librosa.load("/content/gdrive/MyDrive/Audio Classification/clean/"+file, sr = 16000)
      label = str(df.at[file, "label"]) # sampled instrument
    except:
      print("can't read file; moving onto next file")
      continue
    try:

      rand_index = np.random.randint(0, wav.shape[0]-step)
      sample = wav[rand_index:rand_index+step] # sample for about 0.7 seconds
      X_sample = librosa.feature.melspectrogram(y = sample, sr = sr, n_fft = 1024, hop_length = 128, n_mels = 128)
      X_sample = normalize_mel(librosa.power_to_db(X_sample, ref = np.max)) # Normalized Log Mel Spectrogram
      X.append(X_sample)
      label = np.where(classes == label)[0][0]
      labels.append(label)

    except:

      continue

  return X, labels


X, y = build_rand_feat()
X = np.array(X)
y = np.array(y)

np.save("/content/gdrive/MyDrive/Audio Classification/models/Xmel_torch.npy", X)
np.save("/content/gdrive/MyDrive/Audio Classification/models/ymel_torch.npy", y)

