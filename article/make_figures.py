from pathlib import Path
import csv
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

out=Path(__file__).parent/'figures'
manifest=Path(__file__).resolve().parents[1]/'results/audit/manifest.csv'
with manifest.open(newline='') as stream:
    rows=list(csv.DictReader(stream))
plt.rcParams.update({'font.size':8})
fig,ax=plt.subplots(figsize=(3.5,2.7),layout='constrained')
classes=('cyst','normal','stone','tumor')
labels=('Quiste','Normal','Cálculo','Tumor')
left=[0]*4
for split,name,color in [('train','Entrenamiento','#234d70'),('test','Prueba','#568e9a'),('val','Validación','#bdc9c5'),('duplicate','Copias retiradas','#d4ad76')]:
 counts=Counter(r['class'] for r in rows if r['split']==split)
 vals=[counts[c] for c in classes]
 ax.barh(labels,vals,left=left,label=name,color=color)
 left=[x+y for x,y in zip(left,vals)]
ax.set(xlabel='Imágenes',xlim=(0,3200));ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),ncol=2)
fig.savefig(out/'dataset_split.png',dpi=220);plt.close(fig)
fig,ax=plt.subplots(figsize=(7.2,3.0),layout='constrained')
ax.set(xlim=(-0.05,7.2),ylim=(0,3));ax.axis('off')
networks=[('CNN de referencia', ['RGB\n224 × 224\n/ 255','Conv 32\n5×5 · s2\nMaxPool','Conv 64\n5×5 · s2\nMaxPool','Conv 128\n3×3 · s2\nMaxPool','Conv 256\n3×3 · s2\nMaxPool','Flatten\nDense 128\nlineal','Dense 4\nsoftmax']),('YOLO26n-cls', ['RGB\n224 × 224\n/ 255','Conv\nreducción\nespacial','C3k2\nrasgos\nintermedios','Conv y\nC3k2\nprofundos','C2PSA\natención','Classify\nConv y\npromedio','Lineal 4\nsoftmax\ninferencia'])]
for row,(title,boxes) in enumerate(networks):
 y=2.15-row*1.5
 ax.text(0,y+.55,title,fontsize=10,fontweight='bold')
 for column,text in enumerate(boxes):
  x=column*1.03
  ax.add_patch(FancyBboxPatch((x,y-.4),.9,.8,boxstyle='round,pad=0.02',facecolor='#edf1f4',edgecolor='#234d70'))
  ax.text(x+.45,y,text,ha='center',va='center',fontsize=7.5)
  if column<6:ax.annotate('',xy=(x+1.01,y),xytext=(x+.92,y),arrowprops={'arrowstyle':'->','color':'#234d70'})
fig.savefig(out/'architectures.png',dpi=260);plt.close(fig)
