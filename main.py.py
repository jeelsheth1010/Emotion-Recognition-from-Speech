#!/home/jeel1012/pyhton_env/env1/bin/python3
import numpy as np
from pydub import AudioSegment
from python_speech_features import mfcc
#from python_speech_features import delta
#from python_speech_features import logfbank
import scipy.io.wavfile as wav
import os
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

np.random.seed(200)




def get_params():
    #available_emotions = ['ang', 'exc', 'fru', 'neu', 'sad']
    available_emotions = ['ang', 'exc', 'sad','neu']
    path_to_samples = 'IEMOCAP_full_release/Samples/'
    conf_matrix_prefix = 'iemocap'
    framerate = 44100
    return available_emotions, '', '', '', path_to_samples, conf_matrix_prefix, framerate, '', 0
  

available_emotions, path_to_wav, path_to_transcription, path_to_labels, path_to_samples, conf_matrix_prefix, framerate, labels_file, label_pos = get_params()



types = {1: np.int8, 2: np.int16, 4: np.int32}

sessions = ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']

def get_transcriptions(path_to_transcriptions, filename):
  f = open(path_to_transcriptions + filename, 'r').read()
  f = np.array(f.split('\n'))
  transcription = {}

  for i in range(len(f) - 1):
    g = f[i]
    i1 = g.find(': ')
    i0 = g.find(' [')
    ind_id = g[:i0]
    ind_ts = g[i1+2:]
    transcription[ind_id] = ind_ts
  return transcription

def split_wav(a,path,p1):
  name=a["id"]
  #print name
  t1=a["start"]*1000
  t2=a["end"]*1000
  newAudio = AudioSegment.from_wav(path+".wav")
  newAudio = newAudio[t1:t2]
  newAudio.export("audio files/"+name+'.wav', format="wav")


def get_emotions(path_to_emotions, filename):
  f = open(path_to_emotions + filename, 'r').read()
  f = np.array(f.split('\n'))
  idx = f == ''
  idx_n = np.arange(len(f))[idx]
  emotion = []
  for i in range(len(idx_n) - 2):
    g = f[idx_n[i]+1:idx_n[i+1]]
    head = g[0]
    #i0 = head.find(' - ')
    start_time = float(head[head.find('[') + 1:head.find(' - ')])
    end_time = float(head[head.find(' - ') + 3:head.find(']')])
    actor_id = head[head.find(filename[:-4]) + len(filename[:-4]) + 1:head.find(filename[:-4]) + len(filename[:-4]) + 5]
    emo = head[head.find('\t[') - 3:head.find('\t[')]
    vad = head[head.find('\t[') + 1:]
    v = float(vad[1:7])
    a = float(vad[9:15])
    d = float(vad[17:23])
    
    emotion.append({'start':start_time, 
                    'end':end_time,
                    'id':filename[:-4] + '_' + actor_id,
                    'v':v,
                    'a':a,
                    'd':d,
                    'emotion':emo})
  
  return emotion


def read_iemocap_data():
  data = []
  for session in sessions:
    print (session)
    path_to_wav = 'IEMOCAP_full_release/' + session + '/dialog/wav/'
    path_to_emotions = 'IEMOCAP_full_release/' + session + '/dialog/EmoEvaluation/'
    path_to_transcriptions = 'IEMOCAP_full_release/' + session + '/dialog/transcriptions/'
    path='IEMOCAP_full_release/' + session + '/wav/'
    files = os.listdir(path_to_wav)
    f1=[]
    c1=0
    #count=0
    for m in files:
        if m.startswith('._'):
            q=0
        else:
            f1.append(m)
            c1=c1+1
    files = [f[:-4] for f in f1]
    
    for f in files:
      
      emotions = get_emotions(path_to_emotions, f + '.txt')
      transcriptions = get_transcriptions(path_to_transcriptions, f + '.txt')
      for ie, e in enumerate(emotions):
        if e['emotion'] in available_emotions:
          e['transcription'] = transcriptions[e['id']]
          split_wav(e,path_to_wav+f,path+f+'/')
          data.append(e)
  return data




def to_categorical(y):
  y_cat = np.zeros((len(y), len(available_emotions)), dtype=int)
  for i in range(len(y)):
    y_cat[i, :] = np.array(np.array(available_emotions) == y[i], dtype=int)

  return y_cat

def save_sample(x, y, name):
  with open(name, 'wb') as csvfile:
    w = csv.writer(csvfile, delimiter=',',quoting=csv.QUOTE_ALL)
    #print "done"
    for i in range(x.shape[0]):
      row = x[i, :].tolist()
      row.append(y[i])
      w.writerow(row)
    #w.close()

def load(name,p):
  with open(name) as csvfile:
    r = csv.reader(csvfile)
    x = []
    y = []
    for row in r:
      #print row
      x.append(row[:-1])
      y.append(row[-1])
  return np.array(x, dtype=float), np.array(y)

def calculate_features(d):
    (rate,sig) = wav.read("audio files/"+d["id"]+".wav")
    mfcc_feat = mfcc(sig, rate, 0.025, 0.01, 13, 26, 512, 0, None, 0.97, 22, True, np.hamming)
    return mfcc_feat	

def get_features(data, save=True, path=path_to_samples):
  #failed_samples = []
  c=0
  #c1=0
  for di, d in enumerate(data):
    if di%1000 == 0: 
      print (di, ' out of ', len(data))
      #print "feature", d['signal']
    st_features =calculate_features(d)
    #print st_features
    #c1=c1+len(st_features)
    x = []
    y = []
    
    for f in st_features:
      if f[1] > 1.e-4:
        c=c+1
        x.append(f)
        y.append(d['emotion'])
    
    x = np.array(x, dtype=float)
    y = np.array(y)

    if save:
      save_sample(x, y, path + d['id'] + '.csv')
  return x, y

def get_field(data, key):
  return np.array([e[key] for e in data])


def get_sample(idx, path, p):
  tx = []
  ty = []
  tt=[]
  c=1
  for i in idx:
    g=str(path+ i + '.csv')
    x, y = load(g,p)
    if len(y)==0:
         print ("##")
         tt.append(c)
         #print g
         continue
    x=np.mean(x,axis=0)
    for i in y:
       n=i
    y=n
    #print "^^^^"
    #if c==10:
	#c=20  
    tx.append(np.array(x, dtype=float))
    ty.append(y)
    c=c+1
  #print tt
  tx = np.array(tx)
  ty = np.array(ty)
  return tx, ty



h=0
data = np.array(read_iemocap_data())
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																		
print ("data")
#print data
print (len(data))
ids = get_field(data, 'id')
#exit(0)
emotions = get_field(data, 'emotion')
print (np.unique(emotions))

for i in range(len(available_emotions)):
  print (available_emotions[i], emotions[emotions == available_emotions[i]].shape[0])
print ("#######")
parts = 5
z=np.random.RandomState(1234)
permutation = z.permutation(len(data))
permuted_ids = ids[permutation]
step = len(data) / parts

preds = []
trues = []
print ("@@@")
X,Y=get_features(data)
#exit(0)
#print X,Y
print ("$$$$$$")

clf = SVC(kernel='rbf')
for part in range(parts):
  h=h+1
  print ("Stackcount:",h)																																																																																																																																																									
  i0 = step * part
  i1 = step * (part + 1)

  train_idx = np.append(permuted_ids[:i0], permuted_ids[i1:])
  test_idx = permuted_ids[i0:i1]
  
  train_x, train_y = get_sample(train_idx, path_to_samples, "train")
  print ("shape", train_x.shape)
  print ("shape", train_y.shape)
  #print train_y
  #print len(train_x)
  #print len(train_y)
  #print len(train_idx)
  #exit(0)
  for i in range(len(test_idx)):
      if(i==141):
          print (test_idx[i])
            
  print ("@@")
  print (len(test_idx))
  
  test_x, test_y = get_sample(test_idx, path_to_samples, "test")
  #print train_X
  sc_X = StandardScaler()
  train_x = sc_X.fit_transform(train_x)
  test_x = sc_X.transform(test_x)
  #print train_x
  ts = 32
  print (len(train_x))
  accur_lin=[]
  print (train_x.shape)	
  print (train_y)
  print (test_y)
  print ("((((((")
  train_y_cat = to_categorical(train_y)
  test_y_cat = to_categorical(test_y)
  print (train_y_cat)
  print (test_y_cat)
  print ("$$$")
  #print train_y_cat.shape
  print("training SVM ") #train SVM
  clf.fit(train_x, train_y)
  print ("fit done")
  print("getting accuracies") #Use score() function to get accuracy
  pred_lin = clf.score(test_x, test_y)
  print ("poly: ", pred_lin)
  accur_lin.append(pred_lin) #Store accuracy in a list

print("Mean value svm: %s" %np.mean(accur_lin)) 
exit(0)
  	
