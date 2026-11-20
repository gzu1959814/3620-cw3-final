"""NB3 supplemental inference-only evaluation, matching NB2 section 6.3.

Source methodology: NB2.ipynb section 6.3, feature-based generative
precision/recall (k=5), 2000 real MNIST test images, 2000 fake images,
fixed torch noise seed 999, CNN checkpoint results/mnist_classifier.pt.
This module is shared supporting code; the teaching NB3 describes each step.
"""
from pathlib import Path
import torch
from torch import nn
import numpy as np

LATENT_DIM = 100
LEVELS = (0.25,0.5,1.0,1.5,2.0)
SEEDS = (42,123,456)
FOLDERS = {0.25:'experiment1',0.5:'experiment2',1.0:'baseline',1.5:'experiment3',2.0:'experiment4'}

class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
            nn.Flatten(),nn.Linear(1568,128),nn.ReLU())
        self.output=nn.Linear(128,10)
    def forward(self,x):
        h=self.features(x)
        return self.output(h),h

def make_generator():
    return nn.Sequential(
        nn.Linear(100,32),nn.LeakyReLU(.2),nn.Dropout(.3),
        nn.Linear(32,64),nn.LeakyReLU(.2),nn.Dropout(.3),
        nn.Linear(64,128),nn.LeakyReLU(.2),nn.Dropout(.3),
        nn.Linear(128,784),nn.Tanh())

def load_generator(results_dir,level,seed,lr_variant=False):
    root=Path(results_dir)
    if lr_variant:root=root/'sensitivity_lr_0002'
    checkpoint=torch.load(root/FOLDERS[float(level)]/f'seed_{seed}.pt',map_location='cpu',weights_only=True)
    model=make_generator()
    model.load_state_dict(checkpoint['generator']);model.eval()
    return model

def feature_extractor(results_dir):
    model=MNISTClassifier()
    model.load_state_dict(torch.load(Path(results_dir)/'mnist_classifier.pt',map_location='cpu',weights_only=True))
    model.eval()
    return model.features.eval()

def get_real_features(data_dir,feature_net,num_samples=2000):
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    from torch.utils.data import DataLoader
    ds=MNIST(root=str(data_dir),train=False,download=False,transform=ToTensor())
    loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0)
    output=[];count=0
    with torch.inference_mode():
        for img,_ in loader:
            output.append(feature_net(img));count+=img.shape[0]
            if count>=num_samples:break
    return torch.cat(output)[:num_samples]

def get_fake_features(generator,feature_net,level,num_samples=2000,batch_size=128):
    rng=torch.Generator().manual_seed(999)
    output=[]
    generator.eval();feature_net.eval()
    with torch.inference_mode():
        for i in range(0,num_samples,batch_size):
            n=min(batch_size,num_samples-i)
            z=(torch.rand(n,100,generator=rng)*2-1)*float(level)
            images=((generator(z).reshape(-1,1,28,28)+1)/2)
            output.append(feature_net(images))
    return torch.cat(output)

def calculate_radii(features,k=5,batch_size=256):
    features=features.detach().cpu().float();out=[]
    for i in range(0,len(features),batch_size):
        end=min(i+batch_size,len(features))
        distances=torch.cdist(features[i:end],features)
        distances[torch.arange(end-i),torch.arange(i,end)]=float('inf')
        out.append(distances.topk(k,largest=False).values[:,-1])
    return torch.cat(out)

def calculate_coverage(queries,references,radii,batch_size=256):
    queries=queries.detach().cpu().float();references=references.detach().cpu().float()
    covered=0
    for i in range(0,len(queries),batch_size):
        dist=torch.cdist(queries[i:i+batch_size],references)
        covered+=(dist<=radii.unsqueeze(0)).any(dim=1).sum().item()
    return covered/len(queries)

def calculate_precision_recall(real_features,fake_features,k=5,real_radii=None):
    if real_radii is None:real_radii=calculate_radii(real_features,k)
    fake_radii=calculate_radii(fake_features,k)
    p=calculate_coverage(fake_features,real_features,real_radii)
    r=calculate_coverage(real_features,fake_features,fake_radii)
    return float(p),float(r)

def evaluate_fixed(results_dir,data_dir,num_samples=2000,seeds=SEEDS,levels=LEVELS):
    features=feature_extractor(results_dir)
    real=get_real_features(data_dir,features,num_samples=num_samples)
    real_radii=calculate_radii(real)
    out=[]
    for seed in seeds:
        generator=load_generator(results_dir,1.0,seed)
        for level in levels:
            fake=get_fake_features(generator,features,level,num_samples=num_samples)
            p,r=calculate_precision_recall(real,fake,real_radii=real_radii)
            out.append({'truncation':level,'seed':seed,'precision':p,'recall':r,'experiment':'fixed_generator','n_samples':num_samples})
            print(f'fixed baseline seed={seed}, inference t={level}: precision={p:.4f} recall={r:.4f}',flush=True)
    return out

def recompute_all(results_dir,data_dir,output_dir,num_samples=2000):
    """Rebuild all NB3 tables with the NB2 feature metric, without GAN retraining.

    NB1/NB2 own full GAN training; the full NB3 ANALYSIS runs 45 evaluations
    (15 separate trained main, 15 sensitivity, 15 fixed-baseline).
    """
    import csv
    outdir=Path(output_dir);outdir.mkdir(exist_ok=True,parents=True)
    extractor=feature_extractor(results_dir)
    real=get_real_features(data_dir,extractor,num_samples)
    rr=calculate_radii(real)
    rows_by_kind={}
    for kind in ('main','sensitivity','fixed'):
        rows=[]
        for level in LEVELS:
            for seed in SEEDS:
                gen=load_generator(results_dir,1.0 if kind=='fixed' else level,seed,lr_variant=(kind=='sensitivity'))
                fake=get_fake_features(gen,extractor,level,num_samples)
                p,r=calculate_precision_recall(real,fake,real_radii=rr)
                rows.append({'truncation':level,'seed':seed,'precision':p,'recall':r})
        filename={'main':'nb2_main_seed_results.csv','sensitivity':'nb2_sensitivity_seed_results.csv','fixed':'nb3_fixed_generator_results.csv'}[kind]
        with (outdir/filename).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        rows_by_kind[kind]=rows
    return rows_by_kind
