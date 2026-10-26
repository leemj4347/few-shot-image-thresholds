# Overall work-unit progress
from pathlib import Path
import datetime,json,threading,time,os,argparse


class Overall:
    def __init__(self,out,cfg,build_only=False):
        self.lock=threading.RLock()
        self.last={}
        tasks=tests=0
        for d in cfg.get("domains",{}).values():
            st=len(d.get("structures",["random"]))
            seeds=len(cfg.get("seeds",[]))
            train=len(cfg.get("train_M",d.get("M",[])))*st*seeds
            other=len(d.get("M",[]))*st*seeds
            tasks+=train+2*other
            tests+=other
        families=2*len(cfg.get("alphas",[]))
        units={"prepare":sum(bool(p.get(k)) for p in cfg.get("pools",{}).values() for k in ("cache","variants","variants2")),
               "pool_load":len(cfg.get("pools",{})),"task_build":tasks}
        if not build_only:
            units.update(baselines=4,training=families*15,calibration=families,
                         evaluation=2*tests,aggregate=3,save=1,verification=1)
        self.data={"schema":1,"status":"running","started":datetime.datetime.now().isoformat(timespec="seconds"),
          "stage":"prepare","model":None,"basis":"planned work units",
          "build_only":build_only,"invalid_tasks":0,"buckets":{k:{"total":v,"done":0,"skipped":0} for k,v in units.items()}}
        self.write()
    def write(self):
        pass
    def advance(self,bucket,count=1,skipped=False,**fields):
        with self.lock:
            b=self.data["buckets"].get(bucket)
            if b is not None:
                n=max(0,min(int(count),b["total"]-b["done"]))
                b["done"]+=n
                if skipped:b["skipped"]+=n
            self.data.update(stage=bucket,**fields)
            self.write()
    def close(self,bucket,skipped=False):
        b=self.data["buckets"][bucket]
        self.advance(bucket,b["total"]-b["done"],skipped=skipped)
    def delta(self,bucket,key,value,**fields):
        value=int(value)
        token=(bucket,key)
        prior=self.last.get(token,0)
        if value>prior:self.advance(bucket,value-prior,**fields)
        self.last[token]=max(prior,value)
    def event(self,stage,fields):
        mode=self.data.get("model")
        if stage=="pool_load":self.delta("pool_load","all",fields.get("completed",0))
        elif stage=="task_build":
            token=("invalid",fields.get("domain"),fields.get("role"))
            current=int(fields.get("invalid",0))
            prior=self.last.get(token,0)
            self.data["invalid_tasks"]+=max(0,current-prior)
            self.last[token]=max(prior,current)
            self.delta("task_build",(fields.get("domain"),fields.get("role")),fields.get("completed",0))
        elif stage=="baselines":self.delta("baselines",(mode,fields.get("step")),1)
        elif stage=="calibrate":self.delta("calibration",mode,fields.get("completed",0))
        elif stage=="evaluate":self.delta("evaluation",mode,fields.get("completed",0))
        with self.lock:
            if stage not in ("task_build","task_cache","evaluate"):
                for k in ("domain","role","M","seed","task_status"):self.data.pop(k,None)

            self.data.update(
                stage=stage,
                **{k:v for k,v in fields.items() if k in ("domain","role","M","seed","task_status","alpha","n")},
            )
            self.write()
    def state(self,status):
        with self.lock:
            self.data["status"]=status
            self.write()


def finalize(out, failed=False):
    pass
