import tensorflow as tf
import numpy as np

train=False
def to_one_hot(indices, depth):
    idx=np.asarray(indices, dtype=int)
    return np.eye(depth)[idx]
def load_data(path, num_jobs):
    data=np.loadtxt(path, delimiter=',')
    X=data[:, :-1]
    y_idx=data[:, -1].astype(int)
    if data.shape[0]!=0:
        y=to_one_hot(y_idx, num_jobs)
    else:
        y=np.empty((0, num_jobs))
    return X, y
def normalise(x, e=0.01):
    min_val=x.min()
    max_val=x.max()
    normali=(x-min_val)/(max_val-min_val)
    normali=normali*(1-2*e)+e
    return normali
def transpose(matrix):
    return list(map(list, zip(*matrix)))
job=["Strażak", "Policjant", "Ratownik medyczny", "Żołnierz", "Data scientist", "Data analyst", "Business analyst", "Specjalista ds cyberbezpieczeństwa", 
    "Programista", "Statystyk", "Matematyk", "Fizyk", "Kierownik projektu", "Grafik komputerowy", "Projektant UI/UX", "Pracownik HR", "Konsultant IT", 
    "Tester oprogramowania", "Trener personalny", "Ogrodnik", "Lekarz", "Weterynarz", "Farmaceuta", "Stomatolog", "Pielęgniarka", "Fizjoterapeuta", 
    "Psycholog", "Psychiatra", "Coach", "Hydraulik", "Stolarz", "Mechanik", "Elektryk", "Kierownik magazynu", "Operator maszyn", "Projektant wnętrz", "Urbanista", 
    "Kucharz", "Kelner", "Reżyser", "Aktor", "Muzyk", "Fotograf", "Pisarz", "Architekt", "Montażysta wideo", "Realizator dźwięku", "Kierownik sklepu", 
    "Przedstawiciel handlowy", "Specjalista ds marketingu", "Specjalista ds social mediów", "Tłumacz", "Doradca_podatkowy", "Prawnik", "Dziennikarz", 
    "Analityk finansowy", "Księgowy", "Agent ubezpieczeniowy", "Pracownik biura podróży", "Pilot samolotu", "Maszynista", "Kierowca", "Logistyk", "Nauczyciel", 
    "Bibliotekarz", "Naukowiec", "Fryzjer", "Kosmetyczka", "Przedsiębiorca", "Komornik"]

try:
    X_train, y_train=load_data('data/inout/xy_t.csv', len(job))
except:
    np.random.seed(42)
    X_train=np.random.rand(20, 25)
    y_indices=np.random.randint(0, len(job), size=20)
    y_train=to_one_hot(y_indices, len(job))

model=tf.keras.Sequential([
    tf.keras.layers.Dense(64, input_shape=(25,)),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dense(32),
    tf.keras.layers.LeakyReLU(alpha=0.01),
    tf.keras.layers.Dense(len(job), activation='sigmoid') #score per job
])
try:
    model.load_weights("data/.weights.h5")
except:
    pass

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
if train:
    model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)
    model.save_weights("data/.weights.h5")

def recommendations(personality_vector, wpep_mode, job_count):
    scores=model.predict(np.array([personality_vector]))[0]
    top_indices=np.argsort(scores)[-job_count:][::-1]
    recom=[(job[i], scores[i]) for i in top_indices]
    trecom=transpose(recom)
    outscores=trecom[1]
    if wpep_mode!=0:
        if wpep_mode==1:
            income=np.loadtxt("/home/dave/hh/data/inout/wpep.csv", delimiter=',')
        if wpep_mode==2:
            income=np.loadtxt("/home/dave/hh/data/inout/wpep5years.csv", delimiter=',')
        tincome=[income[i] for i in top_indices]
        nincome=normalise(np.array(tincome))
        out=outscores-(1-nincome)
        nout=normalise(out)
        return trecom[0], nout
    else:
        return trecom[0], trecom[1]