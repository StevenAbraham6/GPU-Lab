from .util import fingerprint

def flatten(run):
    rows=[]
    for result in run.get("results",[]):
        for case in result.get("cases",[]):
            for m in case.get("metrics",[]):
                if case.get("status")!="passed":continue
                rows.append(dict(test_id=result["test_id"],case=case["name"],settings=case.get("settings",{}),
                                 power_scope=case.get("power",{}).get("power_scope"),**m))
    return rows

def comparison(a,b):
    """Conservative pairwise comparison; mismatches stay visible without a winner."""
    rows=[];brows={(r["test_id"],r["case"],r["name"],r["unit"]):r for r in flatten(b)}
    warnings=[]
    if a.get('simulated') or b.get('simulated'):warnings.append('SIMULATED demonstration values are present; these are not hardware measurements')
    for field in ('duration','warmup','repeats','sample_interval'):
        if a.get('config',{}).get(field)!=b.get('config',{}).get(field):warnings.append(f'Different {field}; review repeatability and measurement conditions')
    for label,key in (("OS/kernel","host"),("application version","app_version")):
        if a.get("snapshot",{}).get(key)!=b.get("snapshot",{}).get(key):warnings.append(f"Different {label}; inspect detected configuration")
    for left in flatten(a):
        key=(left["test_id"],left["case"],left["name"],left["unit"]);right=brows.pop(key,None)
        reasons=[]
        if right is None:reasons.append("Matching result missing")
        if a.get("simulated")!=b.get("simulated"):reasons.append("Simulated and hardware data cannot be ranked")
        if a.get("app_version")!=b.get("app_version"):reasons.append("Different suite versions")
        # Workload settings must match; vendor/version/preset fields are contextual differences.
        ignored={"torch_version","library","encoder","preset","mode_note"}
        ls={k:v for k,v in left["settings"].items() if k not in ignored}
        rs={k:v for k,v in right["settings"].items() if k not in ignored} if right else {}
        if right and ls!=rs:reasons.append("Workload settings differ")
        if right and left["direction"]!=right["direction"]:reasons.append("Metric direction differs")
        energy=any(s in left["name"] for s in ("power","joule","energy"))
        if energy and right and left.get("power_scope")!=right.get("power_scope"):
            reasons.append("Different GPU power sensor domains; efficiency comparison requires scope review")
        if left["test_id"]=="video" and left["name"] in ("PSNR","SSIM","VMAF","encode throughput","work per joule") and right:
            # Equal target bitrate isn't necessarily equal achieved bitrate or output quality.
            reasons.append("Video quality/bitrate tradeoff: review quality and actual bitrate together")
        improvement=None
        if right and not reasons and left["direction"] in ("higher","lower"):
            av,bv=left["value"],right["value"]
            if av>0:improvement=(bv-av)/av*100*(1 if left["direction"]=="higher" else -1)
            else:reasons.append('Baseline is zero; percentage change is undefined')
        rows.append(dict(test_id=left["test_id"],case=left["case"],metric=left["name"],unit=left["unit"],direction=left["direction"],baseline=left["value"],candidate=right["value"] if right else None,
                         improvement_pct=improvement,reasons=reasons))
    for right in brows.values():
        rows.append(dict(test_id=right["test_id"],case=right["case"],metric=right["name"],unit=right["unit"],direction=right["direction"],baseline=None,candidate=right["value"],improvement_pct=None,reasons=["Baseline result missing"]))
    return dict(baseline=a["id"],candidate=b["id"],warnings=warnings,rows=rows)
