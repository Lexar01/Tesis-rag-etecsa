# %%
import os
from pypdf import PdfReader

pdf_folder = "./dataset"
output_file = "Texto entrenamiento.txt"
all_text = ""

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        reader = PdfReader(pdf_path)

        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"

with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_text)

print(f"Archivo creado: {output_file}")


# %%
with open("Texto entrenamiento.txt", "r", encoding="utf-8") as file:
    dt = file.readlines()

dt


# %%
for i in range(len(dt)):
    dt[i]= dt[i].lower().replace('\n','')

dt

# %%
sum = 0
for i in range(len(dt)):
    sum+=len(dt[i].split())

print(sum)

# %%
dt_splited = []
for i in range(len(dt)):
    dt_splited.append(dt[i].split())

dt_splited

# %%
bigrams= []

for words in dt_splited:
    for i in range(len(words)-1):
        for j in range(i+1,len(words)):
            bigrams.append([words[i],words[j]])
            bigrams.append([words[j],words[i]])

bigrams

# %%
voc = []
vc = []
for i in range(len(dt)):
    for j in range(len(dt[i].split())):
        aux = dt[i].split()
        voc.append(aux[j].split())

for i in range(len(voc)):
    vc.append(voc[i][0])

vc


# %%
vocabulary = []
for word in vc:
    if word not in vocabulary:
        vocabulary.append(word)
vocabulary

# %%
len(vocabulary)

# %%
enc={}

counter=0
for word in vocabulary:
    enc[word] = counter
    counter+=1

enc

# %%
import numpy as np

onehot = np.zeros((len(vocabulary), len(vocabulary)))

for i in range (len(vocabulary)):
    onehot[i][i]=1

onehot

# %%
onehot_dict= {}

for i in range(len(vocabulary)):
    onehot_dict[vocabulary[i]]=onehot[i]

for word in onehot_dict:
    print(word,":",onehot_dict[word])

# %%
x=[]
y=[]

for bi in bigrams:
    x.append(enc[bi[0]])
    y.append(enc[bi[1]])

x= np.array(x)
y= np.array(y)

print(x)
print(y)

# %%
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, Dense, Flatten 

model= Sequential([
    Embedding(input_dim=len(vocabulary), output_dim=200, input_length=1),
    Flatten(),
    Dense(len(vocabulary), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# %%
model.fit(x,y,epochs=10, batch_size=256)

# %%
word= "proyectos"
indice = enc[word]

pred= model.predict(np.array([indice]))
print(pred.shape)
      
predicted_index = np.argmax(pred)
for w, idx in enc.items():
    if idx==predicted_index:
        print("palabra de contexto predicha", w)

print(predicted_index)

# %%
model.save("modelo_skipgram.keras")

# %%
import json

# Al entrenar
with open("enc.json", "w", encoding="utf-8") as f:
    json.dump(enc, f)

with open("vocabulary.json", "w", encoding="utf-8") as f:
    json.dump(vocabulary, f)


# %%
from tensorflow import keras
m= keras.models.load_model("modelo_skipgram.keras")


# %%
m.get_weights()[0]

# %%
w = m.get_weights()[0]

words_emb= {}

for word in vocabulary:
    words_emb[word] = w[enc[word]]

for word in vocabulary:
    print(word,":",words_emb[word])


