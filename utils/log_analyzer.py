import json, statistics
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def summarize(path):
    with open(path) as f:
        d=json.load(f)
    def best(arr):
        m=max(arr)
        return m, arr.index(m)+1
    out={}
    out['epochs']=len(d.get('epoch_loss',[]))
    out['train_acc_best']=best(d.get('epoch_acc',[]))
    out['val_acc_best']=best(d.get('val_acc',[]))
    # domain_test_acc best per domain
    best_domain={}
    for i,entry in enumerate(d.get('domain_test_acc',[])):
        if not entry: continue
        for k,v in entry.items():
            if k not in best_domain or v>best_domain[k][1]:
                best_domain[k]=(i+1,v)
    out['domain_test_best']=best_domain
    # divergence/sharpness stats
    for key in ['divergence_norm','sharpness_var','sharpness_entropy','lambda_balance']:
        arr=d.get(key,[])
        if arr:
            out[key]={
                'min':min(arr),
                'max':max(arr),
                'mean':statistics.mean(arr)
            }
        else:
            out[key]=None
    return out

paths={
    'sasm':os.environ.get(
        'DASM_LOG_ANALYZER_SASM_PATH',
        os.path.join(PROJECT_ROOT, 'models_collection', 'Transformer', 'sasm_train_multi', '20260118_131539_rho0.05_mu0.05_muMode-const_diff-mean', 'train_logs_QIM+PMS+LSB+AHCM_0.5_1s.json')
    ),
    'sam':os.environ.get(
        'DASM_LOG_ANALYZER_SAM_PATH',
        os.path.join(PROJECT_ROOT, 'models_collection', 'Transformer', 'sam_train_AHCM_LSB_PMS_QIM_to_AHCM_LSB_PMS_QIM', 'train_logs_QIM+PMS+LSB+AHCM_0.5_1s.json')
    )
}
for name,path in paths.items():
    print('===',name,'===')
    s=summarize(path)
    for k,v in s.items():
        print(k, v)
