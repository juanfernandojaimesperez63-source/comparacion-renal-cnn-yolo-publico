"""Verifica la adaptación Keras/PyTorch con pesos transferidos, sin entrenar."""
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # La equivalencia se comprueba en CPU.
from pathlib import Path
import numpy as np,torch
from models import ReferenceCNN
from models_mps import ReferenceNetwork
keras=ReferenceCNN(17).model
port=ReferenceNetwork().eval()
convs=[l for l in keras.layers if l.__class__.__name__=='Conv2D']
dense=[l for l in keras.layers if l.__class__.__name__=='Dense']
with torch.no_grad():
 for src,block in zip(convs,port.blocks):
  w,b=src.get_weights(); block[1].weight.copy_(torch.from_numpy(w.transpose(3,2,0,1))); block[1].bias.copy_(torch.from_numpy(b))
 for src,dst in zip(dense,[port.hidden,port.output]):
  w,b=src.get_weights();dst.weight.copy_(torch.from_numpy(w.T));dst.bias.copy_(torch.from_numpy(b))
 x=np.random.default_rng(2).integers(0,256,(3,224,224,3),dtype=np.uint8)
 a=keras(x,training=False).numpy(); b=port(torch.from_numpy(x).permute(0,3,1,2).contiguous().float()/255).softmax(1).numpy()
 np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6)
 count=sum(p.numel() for p in port.parameters());assert count==keras.count_params()==456132
 result={'parameters':count,'max_absolute_probability_difference':float(np.max(np.abs(a-b))),'rtol':1e-5,'atol':1e-6,'passed':True}
 print(json.dumps(result));(Path(__file__).resolve().parents[1]/'docs/architecture_equivalence.json').write_text(json.dumps(result,indent=2))
