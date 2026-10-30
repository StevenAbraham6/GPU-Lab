import csv
import html
import io
import json
from datetime import datetime,timezone
from pathlib import Path
from .catalog import BY_ID
from .compare import flatten
from .util import now

def fmt(v):
    if v is None:return "Unavailable"
    if isinstance(v,(float,int)):
        return f"{v:,.3g}" if abs(v)<.01 and v else f"{v:,.2f}".rstrip("0").rstrip(".")
    return str(v)

def display_metric(value, unit):
    """Scale high-level FP32 throughput for human-facing reports."""
    if unit == 'GFLOP/s':
        return value / 1000, 'TFLOP/s'
    return value, unit

def direction(value):
    return {"higher":"Higher is better","lower":"Lower is better","context":"Context - interpret with workload"}.get(value,"Context")

def representative(result):
    cases=[c for c in result.get('cases',[]) if c.get('status')=='passed' and c.get('metrics')]
    if not cases:return None
    tid=result['test_id']
    if tid=='matrix':
        candidates=[c for c in cases if c.get('settings',{}).get('precision')=='fp32' and c['settings'].get('shape')=='square']
        if candidates:return max(candidates,key=lambda c:c['settings']['n'])
    if tid=='memory':
        candidates=[c for c in cases if c.get('settings',{}).get('operation')=='copy']
        if candidates:return max(candidates,key=lambda c:c['settings']['buffer_bytes'])
    if tid=='transfer':
        candidates=[c for c in cases if c.get('settings',{}).get('direction')=='upload']
        if candidates:return max(candidates,key=lambda c:c['settings']['bytes'])
    return cases[0]

def report_csv(run):
    out=io.StringIO();writer=csv.writer(out)
    writer.writerow(["run_id","run_name","simulated","gpu","test","case","metric","value","unit","direction","settings","power_scope"])
    for row in flatten(run):
        value, unit = display_metric(row["value"], row["unit"])
        cells=[run["id"],run["metadata"]["name"],run["simulated"],run["gpu"]["name"],row["test_id"],row["case"],row["name"],value,unit,direction(row["direction"]),json.dumps(row["settings"],sort_keys=True),row["power_scope"]]
        writer.writerow(["'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v for v in cells])
    return out.getvalue().encode()

def report_html(run):
    e=lambda x:html.escape(str(x))
    meta=run["metadata"];parts=[]
    parts.append(f"<header><div class='eyebrow'>GPU LAB / BENCHMARK REPORT</div><h1>{e(meta['name'])}</h1><p>{e(meta.get('description',''))}</p><div class='meta'>{e(run['gpu']['name'])} · {e(run['status'])} · {e(run['created'])}</div></header>")
    if run["simulated"]:parts.append("<aside>SIMULATED DEMONSTRATION. These values are synthetic; they do not describe any physical GPU.</aside>")
    parts.append(f"<p class='muted'>Run {e(run['id'])} · Suite {e(run['app_version'])} · Metadata revision {run['revision']} · Exported {e(now())}. GPU power only. No overall score.</p>")
    for result in run["results"]:
        test=BY_ID.get(result["test_id"],{});parts.append(f"<section><h2>{e(test.get('title',result['test_id']))} <small>{e(result['status'])}</small></h2>")
        if result.get("reason"):parts.append(f"<p>{e(result['reason'])}</p>")
        for case in result.get("cases",[]):
            parts.append(f"<h3>{e(case['name'])} <small>{e(case['status'])}</small></h3>")
            if case.get("reason"):parts.append(f"<p>{e(case['reason'])}</p>")
            parts.append("<table><thead><tr><th>Metric</th><th>Result</th><th>Interpretation</th></tr></thead><tbody>")
            for m in case.get("metrics",[]):
                value, unit = display_metric(m['value'], m['unit'])
                parts.append(f"<tr><td>{e(m['name'])}</td><td><b>{e(fmt(value))}</b> {e(unit)}</td><td>{e(direction(m['direction']))}</td></tr>")
            parts.append("</tbody></table>")
            if case.get("power",{}).get("reason"):parts.append(f"<p class='muted'>Efficiency: {e(case['power']['reason'])}</p>")
            parts.append(f"<details><summary>How this case was performed</summary><p>{e(case.get('note',''))} {e(case.get('quality_note',''))}</p><pre>{e(json.dumps(case.get('settings',{}),indent=2))}</pre></details>")
        parts.append(f"<details><summary>Methodology and timing boundary</summary><p>{e(test.get('method',''))}</p><p>{e(test.get('boundary',''))}</p></details></section>")
    parts.append(f"<section><h2>Configuration and annotations</h2><p>Edited metadata is shown separately from the original detected evidence.</p><h3>Current metadata</h3><pre>{e(json.dumps(meta,indent=2))}</pre><h3>Original detected configuration</h3><pre>{e(json.dumps(run['snapshot'],indent=2))}</pre><h3>Test configuration</h3><pre>{e(json.dumps(run['config'],indent=2))}</pre><h3>Edit history</h3><pre>{e(json.dumps(run['audit'],indent=2))}</pre></section>")
    css="""body{font:15px/1.6 system-ui,sans-serif;color:#172b3a;max-width:1100px;margin:48px auto;padding:0 28px;background:#fbfcfe}h1{font-size:36px;letter-spacing:-1px}h2{margin-top:0}h3{margin-top:28px}.eyebrow{font-size:12px;font-weight:700;letter-spacing:2px;color:#087e80}.meta,small,.muted{color:#607282}small{font-size:12px}header{border-bottom:3px solid #0a8c8c;padding-bottom:26px}section{background:white;border:1px solid #dce4eb;border-radius:12px;padding:24px;margin:24px 0}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid #e2e9ee;padding:10px}th{background:#edf4f6;font-size:12px;text-transform:uppercase}aside{background:#fff1cf;color:#705015;padding:18px;margin:20px 0;font-weight:700}details{margin:15px 0}summary{cursor:pointer;color:#087e80}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f7fa;padding:15px;font-size:12px}@media print{body{margin:0}section{break-inside:auto}details{display:block}details>*{display:block}}"""
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'><title>GPU Lab report</title><style>"+css+"</style><body>"+"".join(parts)+"</body></html>").encode()

def report_pdf(run):
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,KeepTogether
    from reportlab.graphics.shapes import Drawing,Line,PolyLine,String
    out=io.BytesIO();styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="LabTitle",fontName="Helvetica-Bold",fontSize=26,leading=31,textColor=colors.HexColor("#132c3c"),spaceAfter=16))
    styles.add(ParagraphStyle(name="LabBody",fontName="Helvetica",fontSize=9,leading=13,spaceAfter=8,textColor=colors.HexColor("#253c4b"),wordWrap="CJK"))
    styles.add(ParagraphStyle(name="LabSmall",parent=styles["LabBody"],fontSize=7.5,leading=10,spaceAfter=3))
    styles.add(ParagraphStyle(name="LabHeading",parent=styles["Heading2"],textColor=colors.HexColor("#087f83"),spaceBefore=18,spaceAfter=10))
    def clean(s):
        return str(s).replace("↔","<->").replace("→","->").replace("·"," / ").replace("—","-").replace("–","-").replace("×","x").replace("µ","u").replace("°"," deg ")
    def p(text,style="LabBody"):
        return Paragraph(html.escape(clean(text)),styles[style])
    def table(rows,widths):
        t=Table([[p(c,"LabSmall") for c in row] for row in rows],colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e8f2f3")),("VALIGN",(0,0),(-1,-1),"TOP"),
                               ("LINEBELOW",(0,0),(-1,0),.7,colors.HexColor("#087f83")),("LINEBELOW",(0,1),(-1,-1),.3,colors.HexColor("#dce5eb")),
                               ("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f7f9fb")]),("LEFTPADDING",(0,0),(-1,-1),8),("RIGHTPADDING",(0,0),(-1,-1),8),("TOPPADDING",(0,0),(-1,-1),7),("BOTTOMPADDING",(0,0),(-1,-1),7)]))
        return t
    def plot(points,label,unit):
        d=Drawing(511,175);left,bottom,width,height=48,30,445,125
        start,end=points[0][0],points[-1][0];maximum=max(y for x,y in points)*1.1 or 1
        for fraction in (0,.5,1):
            y=bottom+height*fraction;d.add(Line(left,y,left+width,y,strokeColor=colors.HexColor('#dce5eb'),strokeWidth=.5))
            d.add(String(0,y-3,fmt(maximum*fraction)+' '+unit,fontName='Helvetica',fontSize=7,fillColor=colors.HexColor('#607282')))
        coordinates=[]
        for x,y in points:coordinates += [left+(x-start)/max(end-start,.001)*width,bottom+y/maximum*height]
        d.add(PolyLine(coordinates,strokeColor=colors.HexColor('#087f83'),strokeWidth=1.5))
        d.add(String(left,12,'0 s',fontSize=7));d.add(String(left+width-40,12,fmt(end-start)+' s',fontSize=7))
        d.add(String(left,164,clean(label),fontSize=8,fillColor=colors.HexColor('#18313f')))
        return d
    story=[p("GPU LAB  /  PERFORMANCE & EFFICIENCY","LabHeading"),p(run["metadata"]["name"],"LabTitle"),p(run["metadata"].get("description",""))]
    if run["simulated"]:story += [p("SIMULATED DEMONSTRATION - SYNTHETIC VALUES. NOT HARDWARE MEASUREMENTS.","LabHeading")]
    story += [table([["Run","GPU / environment"],[run["id"],run["gpu"]["name"]],["Status: "+run["status"],"Profile: "+run["config"]["profile"]],["Started: "+run["created"],"Suite: "+run["app_version"]],["Metadata revision: "+str(run["revision"]),"Exported: "+now()]], [245,266]),
              Spacer(1,14),p("Read this report","LabHeading"),p("Results measure the selected GPU with its installed driver and libraries. GPU power only; whole-server power is excluded. Higher throughput and work per joule are better. Lower latency and energy per identical work unit are better. Power and temperature require context. There is no overall score."),
              p("Coverage","LabHeading")]
    coverage=[["Workload / representative case","Result","Status"]]
    for result in run["results"]:
        case=representative(result);m=case['metrics'][0] if case else None
        label=BY_ID.get(result["test_id"],{}).get("title",result["test_id"])+((' / '+case['name']) if case else '')
        if m:
            value, unit = display_metric(m['value'], m['unit'])
            summary = fmt(value)+' '+unit+' / '+direction(m['direction'])
        else:
            summary = 'No valid measurement'
        coverage.append([label,summary,result['status']])
    story += [table(coverage,[255,175,81]),Spacer(1,8),p('Representative cases are selected by workload definition, not the largest score. Full results and precision/size details follow. Compare only matching configurations.','LabSmall')]
    if run.get("error"):story += [p("Run finding: "+run["error"])]
    power=[(s['ts'],s['power_w']) for s in run.get('telemetry',[]) if s.get('power_w') is not None]
    if len(power)>1:
        story += [PageBreak(),p('GPU power over time','LabTitle'),p('Full-run telemetry includes warm-up, measured work and idle gaps. Each case uses its own covered interval for energy integration.'),plot(power,'Device-reported GPU power','W')]
    # One stable group per section; repeat table headers when a group spans pages.
    for result in run["results"]:
        test=BY_ID.get(result["test_id"],{})
        story += [PageBreak(),p(test.get("title",result["test_id"]),"LabTitle"),p(test.get("summary","")),p("Status: "+result["status"])]
        if result.get("reason"):story.append(p(result["reason"]))
        rows=[["Case / metric","Result","Direction"]]
        for case in result.get("cases",[]):
            if not case.get("metrics"):
                rows.append([case["name"],case["status"],case.get("reason","")]);continue
            for i,m in enumerate(case["metrics"]):
                value, unit = display_metric(m["value"], m["unit"])
                rows.append([(case["name"]+" / " if i==0 else "")+m["name"],fmt(value)+" "+unit,direction(m["direction"])])
        if len(rows)>1:story.append(table(rows,[250,120,141]))
        for case in result.get('cases',[]):
            series=case.get('series',[])
            if len(series)>1:
                metrics=case.get('metrics',[])
                metric_unit=metrics[0].get('unit','') if metrics else ''
                factor=0.001 if metric_unit=='GFLOP/s' else 1
                chart_unit='TFLOP/s' if metric_unit=='GFLOP/s' else metric_unit
                story += [Spacer(1,12),plot([(s['elapsed'],s['throughput']*factor) for s in series],case['name']+' - throughput over time',chart_unit)]
        story += [p("Methodology","LabHeading"),p(test.get("method","")),p("Timing boundary: "+test.get("boundary",""))]
        for case in result.get("cases",[]):
            block=[p(case["name"],"LabHeading"),p("Status: "+case["status"])]
            if case.get("note"):block.append(p(case["note"]))
            if case.get("quality_note"):block.append(p(case["quality_note"]))
            if case.get("power",{}).get("reason"):block.append(p("Energy coverage: "+case["power"]["reason"]))
            settings=[["Parameter","Value"]]+[[k,json.dumps(v,ensure_ascii=True) if isinstance(v,(dict,list)) else str(v)] for k,v in case.get("settings",{}).items()]
            if len(settings)>1:block.append(table(settings,[150,361]))
            variability=[]
            for m in case.get("metrics",[]):
                s=m.get("statistics")
                if s:
                    median, unit = display_metric(s['median'], m['unit'])
                    minimum, _ = display_metric(s['minimum'], m['unit'])
                    maximum, _ = display_metric(s['maximum'], m['unit'])
                    variability.append(f"{m['name']}: median {fmt(median)} {unit}; min {fmt(minimum)}; max {fmt(maximum)}; CV {fmt(s['cv_pct'])}%; n={len(s['samples'])}.")
            if variability:block.append(p(" ".join(variability),"LabSmall"))
            story.append(KeepTogether(block))
    story += [PageBreak(),p("Configuration & provenance","LabTitle"),p("Current editable metadata is separated from original detected evidence. Changes do not rewrite measured results.")]
    for label,data in [("Current metadata",run["metadata"]),("Original detected configuration",run["snapshot"]),("Test configuration",run["config"])]:
        story.append(p(label,"LabHeading"))
        # Flatten one nesting level to keep long configuration tables legible.
        rows=[["Field","Value"]]
        def fields(prefix,obj):
            if isinstance(obj,dict):
                for k,v in obj.items():fields(prefix+"."+k if prefix else k,v)
            else:rows.append([prefix,json.dumps(obj,ensure_ascii=True) if isinstance(obj,list) else str(obj)])
        fields("",data);story.append(table(rows,[180,331]))
    story.append(p("Metadata history","LabHeading"))
    for event in run.get("audit",[]):story.append(p(event["at"]+" - "+event.get("action","event"),"LabSmall"))
    def footer(canvas,doc):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#dce5eb"));canvas.line(42,37,553,37)
        canvas.setFont("Helvetica",7);canvas.setFillColor(colors.HexColor("#607282"))
        canvas.drawString(42,25,"GPU Lab | "+run["id"][:8]+" | revision "+str(run["revision"])+(" | SIMULATED" if run["simulated"] else ""))
        canvas.drawRightString(553,25,str(doc.page));canvas.restoreState()
    doc=SimpleDocTemplate(out,pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=52,title=clean(run["metadata"]["name"]),author="GPU Lab")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    return out.getvalue()
