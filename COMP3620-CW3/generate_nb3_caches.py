from pathlib import Path
import sys,csv,statistics,json,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from nb3_evaluation import *
R=Path(__file__).resolve().parent
CACHE=R/'nb3_cache';CACHE.mkdir(exist_ok=True)

def save(name,rows):
    with (CACHE/name).open('w',newline='',encoding='utf-8') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

F=feature_extractor(R/'results')
real=get_real_features(R/'data',F,2000)
real_radii=calculate_radii(real)

def experiment_rows(sensitivity=False):
    out=[]
    for t in LEVELS:
        for seed in SEEDS:
            G=load_generator(R/'results',t,seed,lr_variant=sensitivity)
            fake=get_fake_features(G,F,t,num_samples=2000)
            p,r=calculate_precision_recall(real,fake,real_radii=real_radii)
            out.append({'truncation':t,'seed':seed,'precision':p,'recall':r})
            print('sensitivity' if sensitivity else 'main',t,seed,p,r,flush=True)
    return out

start=time.time()
main=experiment_rows();save('nb2_main_seed_results.csv',main)
sens=experiment_rows(True);save('nb2_sensitivity_seed_results.csv',sens)
fixed=evaluate_fixed(R/'results',R/'data',2000);save('nb3_fixed_generator_results.csv',fixed)

def summarize(rows):
    out=[]
    for level in LEVELS:
        s=[r for r in rows if float(r['truncation'])==level]
        out.append({'truncation':level,
                    'precision_mean':statistics.mean(float(r['precision']) for r in s),
                    'precision_std':statistics.stdev(float(r['precision']) for r in s),
                    'recall_mean':statistics.mean(float(r['recall']) for r in s),
                    'recall_std':statistics.stdev(float(r['recall']) for r in s)})
    return out
for name,rows in [('nb2_main_summary.csv',main),('nb2_sensitivity_summary.csv',sens),('nb3_fixed_generator_summary.csv',fixed)]:
    save(name,summarize(rows))
print('elapsed',time.time()-start)
print('MAIN',summarize(main))
print('SENS',summarize(sens))
print('FIXED',summarize(fixed))
