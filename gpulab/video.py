"""Explicit hardware video paths. No -hwaccel auto and no software fallback."""
import hashlib
import json
import os
import re
import subprocess
import time
import threading
import concurrent.futures
from pathlib import Path
from .catalog import metric
from .measurement import stats
from .util import command

BITRATES={"1280x720":4_000_000,"1920x1080":8_000_000,"3840x2160":25_000_000}

def encoder(gpu,codec):
    return {"h264":"h264","hevc":"hevc","av1":"av1"}[codec] + ("_nvenc" if gpu["vendor"]=="nvidia" else "_vaapi")

def device_args(gpu):
    if gpu["vendor"]=="nvidia":return []
    node=gpu.get("render_node")
    if not node:raise NotImplementedError("No DRM render node for selected GPU")
    return ["-vaapi_device",node]

def cuda_index(gpu):
    index=gpu.get("nvidia_index")
    if index is None or not gpu.get('uuid'):raise NotImplementedError("Missing NVIDIA UUID-to-PCI mapping; rediscover GPU")
    mask=os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is not None:
        for position,item in enumerate(mask.split(',')):
            if item.strip() in (str(index),gpu.get('uuid')):return '0'
        raise NotImplementedError("Selected GPU is not unambiguously visible in CUDA_VISIBLE_DEVICES")
    # Every hardware FFmpeg subprocess is isolated to the selected UUID below.
    return '0'

def ffmpeg_env(gpu):
    env=os.environ.copy()
    if gpu and gpu.get('vendor')=='nvidia':
        cuda_index(gpu)
        env['CUDA_VISIBLE_DEVICES']=gpu['uuid']
    return env

def encode_args(gpu,codec,mode,bitrate):
    args=["-c:v",encoder(gpu,codec),"-b:v",str(bitrate),"-maxrate",str(bitrate),"-bufsize",str(bitrate*2),"-g","60","-bf","0"]
    if gpu["vendor"]=="nvidia":
        args += ["-gpu",cuda_index(gpu),"-preset","p4","-tune","ll" if mode=="low-latency" else "hq","-rc","cbr","-pix_fmt","yuv420p"]
    else:
        # VAAPI uploads are part of encode-only wall time. No cross-vendor preset equivalence claimed.
        args += ["-rc_mode","CBR"]
    return args

def probe_video(gpu,codec):
    enc=encoder(gpu,codec)
    try:
        args=["ffmpeg","-hide_banner","-loglevel","error","-nostdin",*device_args(gpu),"-f","lavfi","-i","testsrc2=size=128x128:rate=30","-frames:v","2"]
        if gpu["vendor"]!="nvidia":args += ["-vf","format=nv12,hwupload"]
        args += encode_args(gpu,codec,"quality",1_000_000)+["-f","null","-"]
        rc,out,err=command(args,timeout=25,env=ffmpeg_env(gpu))
        return dict(encoder=enc,supported=rc==0,reason=None if rc==0 else (err or out)[-1500:],command=args)
    except Exception as e:
        return dict(encoder=enc,supported=False,reason=str(e))

def run_command(args,log,gpu=None):
    start=time.time()
    with log.open("ab") as f:
        f.write(("\nCOMMAND " + json.dumps(args) + "\n").encode());f.flush()
        p=subprocess.run(args,stdout=f,stderr=f,stdin=subprocess.DEVNULL,env=ffmpeg_env(gpu))
    end=time.time()
    if p.returncode:
        raise RuntimeError(log.read_text(errors="replace")[-2500:])
    return start,end

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024**2),b""):h.update(block)
    return h.hexdigest()

def quality(reference,encoded,directory,label):
    log=directory/f"quality-{label}.log"
    args=["ffmpeg","-hide_banner","-nostdin","-i",str(encoded),"-i",str(reference),"-lavfi","[0:v]setpts=PTS-STARTPTS,format=yuv420p[d];[1:v]setpts=PTS-STARTPTS,format=yuv420p[r];[d][r]psnr","-f","null","-"]
    run_command(args,log);text=log.read_text(errors="replace")
    metrics=[];m=re.search(r"average:([0-9.]+|inf)",text)
    if m and m[1]!="inf":metrics.append(metric("PSNR",float(m[1]),"dB","higher"))
    args[args.index("-lavfi")+1]="[0:v]setpts=PTS-STARTPTS,format=yuv420p[d];[1:v]setpts=PTS-STARTPTS,format=yuv420p[r];[d][r]ssim"
    run_command(args,log);text=log.read_text(errors="replace");m=re.search(r"All:([0-9.]+)",text)
    if m:metrics.append(metric("SSIM",float(m[1]),"score","higher"))
    filters=command(["ffmpeg","-hide_banner","-filters"])[1]
    note="VMAF not present in this FFmpeg build; no perceptual quality score."
    if "libvmaf" in filters:
        args[args.index("-lavfi")+1]="[0:v]setpts=PTS-STARTPTS,format=yuv420p[d];[1:v]setpts=PTS-STARTPTS,format=yuv420p[r];[d][r]libvmaf=model=version=vmaf_v0.6.1"
        try:
            run_command(args,log);text=log.read_text(errors="replace");m=re.search(r"VMAF score:\s*([0-9.]+)",text)
            if m:metrics.append(metric("VMAF",float(m[1]),"score","higher"));note="VMAF v0.6.1 explicitly pinned; used for SDR test content only."
        except RuntimeError as e:note="VMAF filter/model execution failed: " + str(e)[-400:]
    return metrics,note

def video(gpu,cfg,directory):
    cases=[];directory=Path(directory);directory.mkdir(exist_ok=True,parents=True)
    for resolution in cfg["resolutions"]:
        reference=directory/f"reference-{resolution}.mkv"
        log=directory/f"prepare-{resolution}.log"
        # A deterministic, freely generated calibration clip is included. Production comparisons should use a shared representative reference.
        args=["ffmpeg","-hide_banner","-loglevel","warning","-nostdin","-y"]
        if cfg.get("reference"):
            args += ["-i",str(Path(cfg["reference"]).resolve()),"-vf",f"scale={resolution.replace('x',':')}:flags=bicubic,fps=30,format=yuv420p"]
        else:
            args += ["-f","lavfi","-i",f"testsrc2=size={resolution}:rate=30","-vf","format=yuv420p"]
        args += ["-t",str(cfg["video_seconds"]),"-an","-c:v","ffv1",str(reference)]
        run_command(args,log)
        rc,out,err=command(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","json",str(reference)],timeout=120)
        if rc:raise RuntimeError(err)
        frames=int(json.loads(out)["streams"][0]["nb_read_frames"])
        if frames<2:raise RuntimeError("Reference clip too short")
        # Hash decoded pixel content, not Matroska's random container identifiers.
        rc,hashed,err=command(["ffmpeg","-v","error","-nostdin","-i",str(reference),"-map","0:v:0","-f","hash","-hash","sha256","-"],timeout=300)
        if rc:raise RuntimeError("Reference pixel hashing failed: "+err)
        ref_hash=hashed.split('=',1)[-1].strip()
        for codec in cfg["codecs"]:
            supported=probe_video(gpu,codec)
            for mode in cfg["video_modes"]:
                settings=dict(codec=codec,resolution=resolution,mode=mode,bitrate=BITRATES[resolution],fps=30,pixel_format="yuv420p",gop=60,b_frames=0,
                              reference_sha256=ref_hash,reference_kind="user clip normalized to SDR 8-bit 4:2:0" if cfg.get("reference") else "synthetic testsrc2 calibration clip (not representative content coverage)",
                              frames=frames,encoder=encoder(gpu,codec),ffmpeg_version=command(["ffmpeg","-version"])[1].splitlines()[0],preset="NVENC p4" if gpu["vendor"]=="nvidia" else "VAAPI driver default",
                              mode_note="NVENC ll/hq tune" if gpu["vendor"]=="nvidia" else "VAAPI mode uses GOP60/B0 CBR in both cases; distinct vendor low-latency tuning not implemented")
                label=f"{codec}-{resolution}-{mode}"
                if not supported["supported"]:
                    cases.append(dict(name=label,status="unsupported",reason=supported["reason"],settings=settings,metrics=[],intervals=[]));continue
                encoded=directory/f"{label}.mkv";elog=directory/f"{label}.log"
                common=["ffmpeg","-hide_banner","-loglevel","info","-nostdin","-y"]
                encode=common+device_args(gpu)+["-i",str(reference),"-an"]
                if gpu["vendor"]!="nvidia":encode += ["-vf","format=nv12,hwupload"]
                encode += encode_args(gpu,codec,mode,BITRATES[resolution])+[str(encoded)]
                try:
                    # Untimed runtime warm-up at actual resolution and mode.
                    warm=encode[:-1]+["-frames:v","2","-f","null","-"];run_command(warm,elog,gpu)
                    intervals=[];fps=[]
                    for _ in range(cfg["repeats"]):
                        start,end=run_command(encode,elog,gpu);intervals.append(dict(start=start,end=end,work=frames,work_unit="frames"));fps.append(frames/(end-start))
                    qmetrics,qnote=quality(reference,encoded,directory,label)
                    metrics=[metric("encode throughput",stats(fps)["median"],"frames/s",statistics=stats(fps)),metric("output bitrate",encoded.stat().st_size*8/(frames/30)/1e6,"Mbit/s","context"),*qmetrics]
                    cases.append(dict(name="encode · "+label,status="passed",settings={**settings,"pipeline":"encode-upload"},metrics=metrics,intervals=intervals,commands=[encode],quality_note=qnote,
                                      note="Bitrate includes container overhead. Quality comparison requires matching bitrate/content. No per-frame latency claimed."))
                    # Confirm frame count by software-decoding the output after timing.
                    rc,info,err=command(["ffprobe","-v","error","-count_frames","-select_streams","v:0","-show_entries","stream=nb_read_frames","-of","json",str(encoded)],timeout=120)
                    if rc or int(json.loads(info)["streams"][0]["nb_read_frames"])!=frames:raise ArithmeticError("Encoded frame count mismatch")
                except Exception as e:
                    if cases and cases[-1].get("name")=="encode · "+label:cases.pop()
                    cases.append(dict(name="encode · "+label,status="failed_validation" if isinstance(e,ArithmeticError) else "error",reason=str(e)[-2000:],settings=settings,metrics=[],intervals=[]));continue
                decode_device=["-hwaccel","cuda","-hwaccel_device",cuda_index(gpu),"-hwaccel_output_format","cuda"] if gpu["vendor"]=="nvidia" else ["-hwaccel","vaapi","-hwaccel_device",gpu["render_node"],"-hwaccel_output_format","vaapi"]
                for pipeline in ("decode","transcode"):
                    args=common+decode_device+["-i",str(encoded),"-an"]
                    if pipeline=="decode":args += ["-vf","hwdownload,format=nv12","-f","null","-"]
                    else:args += encode_args(gpu,codec,mode,BITRATES[resolution])+["-f","null","-"]
                    try:
                        run_command(args,elog,gpu) # runtime probe/warm-up; explicit hardware frames require hardware decode
                        intervals=[];rates=[]
                        for _ in range(cfg["repeats"]):
                            start,end=run_command(args,elog,gpu);intervals.append(dict(start=start,end=end,work=frames,work_unit="frames"));rates.append(frames/(end-start))
                        cases.append(dict(name=pipeline+" · "+label,status="passed",settings={**settings,"pipeline":pipeline,"encoded_input_sha256":sha256(encoded)},metrics=[metric(pipeline+" throughput",stats(rates)["median"],"frames/s",statistics=stats(rates))],intervals=intervals,commands=[args],note="Decode includes CPU readback; transcode retains hardware frames. Input is this GPU's encode output; cross-run decode comparability requires identical encoded input hash."))
                    except Exception as e:
                        cases.append(dict(name=pipeline+" · "+label,status="unsupported",reason=str(e)[-2000:],settings=settings,metrics=[],intervals=[]))
        # Raw references dominate disk usage; source hash and generation method are retained.
        reference.unlink(missing_ok=True)
    return cases

def concurrent_video(gpu,cfg,directory,compute_factory=None,soak=False):
    """A bounded media capacity/mixed workload; all encoders map to the same GPU."""
    codec=cfg['codecs'][0];resolution=cfg['resolutions'][0];mode=cfg['video_modes'][0]
    supported=probe_video(gpu,codec)
    settings=dict(codec=codec,resolution=resolution,mode=mode,bitrate=BITRATES[resolution],fps=30,pixel_format='yuv420p',
                  encoder=encoder(gpu,codec),pipeline='hardware encode with host reference decode and upload')
    if not supported['supported']:
        return [dict(name='video soak' if soak else 'video concurrency',status='unsupported',reason=supported['reason'],settings=settings,metrics=[],intervals=[])]
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    reference=directory/('soak-reference.mkv' if soak else 'concurrency-reference.mkv')
    prepare=['ffmpeg','-hide_banner','-loglevel','warning','-nostdin','-y']
    if cfg.get('reference'):prepare += ['-i',str(Path(cfg['reference']).resolve()),'-vf',f"scale={resolution.replace('x',':')},fps=30,format=yuv420p"]
    else:prepare += ['-f','lavfi','-i',f'testsrc2=size={resolution}:rate=30']
    prepare += ['-t',str(cfg['video_seconds']),'-an','-c:v','ffv1',str(reference)]
    run_command(prepare,directory/'concurrency-prepare.log')
    rc,out,err=command(['ffprobe','-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(reference)],timeout=120)
    if rc:raise RuntimeError(err)
    frames=int(json.loads(out)['streams'][0]['nb_read_frames'])
    args=['ffmpeg','-hide_banner','-nostdin','-y',*device_args(gpu),'-i',str(reference),'-an']
    if gpu['vendor']!='nvidia':args += ['-vf','format=nv12,hwupload']
    args += encode_args(gpu,codec,mode,BITRATES[resolution])+['-f','null','-']
    results=[];single=None
    combinations=[(1,False)] if soak else [(1,False),(2,False),(4,False)]+([(1,True)] if compute_factory else [])
    for streams,mixed in combinations:
        stop=threading.Event();compute_work=[0.0];compute_error=[];compute_thread=None
        try:
            run_command(args,directory/'concurrency-warmup.log',gpu)
            series=[];intervals=[];rates=[];completion=[];begin=time.time();rounds=0
            if mixed:
                fn,work,buf,compute_settings=compute_factory()
                def compute_loop():
                    try:
                        while not stop.is_set():fn();compute_work[0]+=work
                    except Exception as e:compute_error.append(str(e));stop.set()
                compute_thread=threading.Thread(target=compute_loop,daemon=True)
                compute_thread.start()
            with concurrent.futures.ThreadPoolExecutor(max_workers=streams) as pool:
                while rounds==0 or (time.time()-begin<cfg['soak_seconds'] if soak else rounds<cfg['repeats']):
                    start=time.time()
                    futures=[pool.submit(run_command,args,directory/f'concurrency-{streams}-{mixed}-{i}.log',gpu) for i in range(streams)]
                    times=[future.result() for future in futures];end=time.time()
                    completion.extend(e-s for s,e in times);rate=frames*streams/(end-start);rates.append(rate)
                    series.append(dict(elapsed=end-begin,throughput=rate))
                    intervals.append(dict(start=start,end=end,work=frames*streams,work_unit='frames'));rounds+=1
            stop.set()
            if compute_thread:compute_thread.join(timeout=30)
            if compute_error:raise RuntimeError('Concurrent compute failed: '+compute_error[0])
            med=stats(rates)['median']
            if single is None:single=med
            metrics=[metric('aggregate encode throughput',med,'frames/s',statistics=stats(rates)),
                     metric('per-stream slowdown',single/(med/streams),'×','lower'),
                     metric('clip completion time',stats(completion)['median'],'s','lower',statistics=stats(completion))]
            if mixed:metrics += [metric('concurrent compute throughput',compute_work[0]/(time.time()-begin),'GFLOP/s')]
            if soak and len(series)>1:
                q=max(1,len(series)//4);early=sum(s['throughput'] for s in series[:q])/q;late=sum(s['throughput'] for s in series[-q:])/q
                metrics.append(metric('late / early throughput',late/early*100,'%','context'))
            results.append(dict(name='sustained hardware video' if soak else f'{streams} video streams'+(' + FP32 compute' if mixed else ''),status='passed',settings={**settings,'streams':streams,'mixed_compute':mixed,'frames_per_stream':frames,'soak_seconds':cfg['soak_seconds'] if soak else None},
                                metrics=metrics,intervals=intervals,series=series,commands=[args],note='Host reference decoding, hardware uploads and process startup included. Clip completion is not per-frame latency. Shared-GPU energy is attributed to the entire concurrent workload, not individual jobs.'))
        except Exception as e:
            results.append(dict(name='video soak' if soak else f'{streams} concurrent video streams',status='error',reason=str(e)[-2000:],settings=settings,metrics=[],intervals=[]))
        finally:
            stop.set()
            if compute_thread:compute_thread.join(timeout=30)
    reference.unlink(missing_ok=True)
    return results
