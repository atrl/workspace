from __future__ import annotations

import argparse
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from .stage1 import build_stage1_dashboard_payload, resolve_stage1_dashboard_source


APP_HTML = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Stage1 · Qlib 决策台</title>
<style>
:root{--bg:#071018;--panel:#0d1924;--line:#203242;--text:#e9f0f5;--muted:#91a5b5;--cyan:#4dd9d0;--gold:#f4c86a;--red:#ff7777;--blue:#6e9fff}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 80% 0,#173344 0,transparent 35%),var(--bg);color:var(--text);font:14px system-ui,-apple-system,sans-serif}
main{max-width:1440px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;align-items:end;gap:18px;flex-wrap:wrap}
h1{font-size:28px;margin:0 0 5px}.eyebrow,.muted{color:var(--muted)}.controls{display:flex;gap:10px;align-items:end}
label{display:grid;gap:5px;color:var(--muted);font-size:12px}input,button{background:#101f2b;color:var(--text);border:1px solid var(--line);border-radius:8px;padding:9px 11px}
button{cursor:pointer;background:var(--cyan);color:#06201f;font-weight:700}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px;margin-top:18px}
.card{background:linear-gradient(145deg,rgba(17,34,47,.96),rgba(10,22,32,.96));border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 10px 28px #0005}
.kpi{grid-column:span 3}.kpi b{font-size:24px;display:block;margin-top:7px}.wide{grid-column:span 8}.side{grid-column:span 4}.full{grid-column:1/-1}
h2{font-size:15px;margin:0 0 12px}canvas{width:100%;height:300px;display:block}.weights{display:flex;height:18px;border-radius:9px;overflow:hidden;margin:12px 0}
.weights span:nth-child(1){background:var(--cyan)}.weights span:nth-child(2){background:var(--blue)}.weights span:nth-child(3){background:#445565}
.event{padding:11px 0;border-top:1px solid var(--line)}.pill{display:inline-block;border-radius:20px;padding:3px 8px;background:#173240;color:var(--cyan);font-size:12px}
table{width:100%;border-collapse:collapse}th,td{text-align:right;padding:8px;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}.pos{color:var(--cyan)}.neg{color:var(--red)}
@media(max-width:900px){.kpi,.wide,.side{grid-column:1/-1}.controls{width:100%}}
</style></head><body><main>
<div class="top"><div><div class="eyebrow">RECORDER-FIRST / PAPER SIM</div><h1>Stage1 · Qlib 决策台</h1><div class="muted" id="source">正在读取冻结模型证据…</div></div>
<div class="controls"><label>开始<input id="start" type="date"></label><label>结束<input id="end" type="date"></label><button onclick="loadData()">运行区间推理</button></div></div>
<section class="grid">
<div class="card kpi"><span class="muted">Stage1 区间收益</span><b id="ret">—</b><small id="range"></small></div>
<div class="card kpi"><span class="muted">最大回撤</span><b id="dd">—</b><small>区间内重新归一</small></div>
<div class="card kpi"><span class="muted">相对 SOXL</span><b id="vssoxl">—</b><small>收益百分点</small></div>
<div class="card kpi"><span class="muted">相对 SOXX</span><b id="vssoxx">—</b><small>收益百分点</small></div>
<div class="card wide"><h2>净值对比 · Stage1 / SOXL / SOXX</h2><canvas id="nav"></canvas></div>
<div class="card side"><h2>最新冻结模型推理</h2><div id="latest"></div></div>
<div class="card wide"><h2>仓位与买卖点</h2><canvas id="weight"></canvas></div>
<div class="card side"><h2>因子差异 · 买卖点诊断</h2><div class="muted" id="disclaimer"></div><div id="events"></div></div>
<div class="card full"><h2>区间表现</h2><table><thead><tr><th>对象</th><th>收益</th><th>年化</th><th>最大回撤</th><th>波动</th><th>Sharpe</th></tr></thead><tbody id="metrics"></tbody></table></div>
</section></main>
<script>
const colors={Stage1:'#4dd9d0',SOXL:'#f4c86a',SOXX:'#6e9fff'};
const pct=x=>(x??0).toFixed(2)+'%'; const cls=x=>x>=0?'pos':'neg';
function lineChart(id, series, keys){const c=document.getElementById(id),dpr=devicePixelRatio||1,w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);x.clearRect(0,0,w,h);x.strokeStyle='#203242';x.fillStyle='#91a5b5';x.font='11px system-ui';
 let vals=keys.flatMap(k=>series.map(r=>r[k]));let lo=Math.min(...vals),hi=Math.max(...vals);if(hi===lo)hi=lo+1;
 for(let i=0;i<5;i++){let y=12+(h-34)*i/4;x.beginPath();x.moveTo(42,y);x.lineTo(w-8,y);x.stroke();x.fillText((hi-(hi-lo)*i/4).toFixed(2),3,y+4)}
 keys.forEach(k=>{x.strokeStyle=colors[k]||'#fff';x.lineWidth=2;x.beginPath();series.forEach((r,i)=>{let px=42+(w-52)*i/Math.max(1,series.length-1),py=12+(h-34)*(hi-r[k])/(hi-lo);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke()});
 keys.forEach((k,i)=>{x.fillStyle=colors[k]||'#fff';x.fillRect(50+i*100,h-12,10,3);x.fillText(k,65+i*100,h-8)})}
function weightChart(rows,events){lineChart('weight',rows.map(r=>({...r,SOXL:r.soxl_weight,SOXX:r.soxx_weight,Cash:r.cash_weight})),['SOXL','SOXX','Cash']);}
async function loadData(){let q=new URLSearchParams();if(start.value)q.set('start',start.value);if(end.value)q.set('end',end.value);let res=await fetch('/api/stage1?'+q);let d=await res.json();if(!res.ok){alert(d.detail||'加载失败');return}
 start.min=d.available_range.start;start.max=d.available_range.end;end.min=d.available_range.start;end.max=d.available_range.end;if(!start.value)start.value=d.selected_range.start;if(!end.value)end.value=d.selected_range.end;
 source.textContent=`qlib recorder ${d.recorder_id} · ${d.selected_range.rows} 个信号日`;range.textContent=`${d.selected_range.start} → ${d.selected_range.end}`;
 let m=d.metrics;ret.textContent=pct(m.Stage1.return_pct);ret.className=cls(m.Stage1.return_pct);dd.textContent=pct(m.Stage1.max_drawdown_pct);dd.className=cls(m.Stage1.max_drawdown_pct);
 let a=m.Stage1.return_pct-m.SOXL.return_pct,b=m.Stage1.return_pct-m.SOXX.return_pct;vssoxl.textContent=(a>=0?'+':'')+pct(a);vssoxl.className=cls(a);vssoxx.textContent=(b>=0?'+':'')+pct(b);vssoxx.className=cls(b);
 lineChart('nav',d.rows.map(r=>({...r,Stage1:r.stage1_nav,SOXL:r.soxl_nav,SOXX:r.soxx_nav})),['Stage1','SOXL','SOXX']);weightChart(d.rows,d.trade_events);
 let l=d.latest_inference,w=l.weights;latest.innerHTML=`<span class="pill">${l.status}</span><h3>${l.signal_date}</h3><div class="weights"><span style="width:${w.SOXL*100}%"></span><span style="width:${w.SOXX*100}%"></span><span style="width:${w.cash*100}%"></span></div><div>SOXL ${(w.SOXL*100).toFixed(1)}% · SOXX ${(w.SOXX*100).toFixed(1)}% · 现金 ${(w.cash*100).toFixed(1)}%</div><p class="muted">组合回撤 ${pct(l.portfolio_drawdown*100)} · beta cap ${l.current_beta_cap.toFixed(2)} · 恢复动作 ${l.recovery_action?'是':'否'}</p>`;
 disclaimer.textContent=d.factor_disclaimer;events.innerHTML=d.trade_events.slice(-8).reverse().map(e=>`<div class="event"><b>${e.trade_date} · ${e.direction}</b><div class="muted">SOXL ${e.soxl_delta>=0?'+':''}${pct(e.soxl_delta*100)} · SOXX ${e.soxx_delta>=0?'+':''}${pct(e.soxx_delta*100)}</div>${e.factors.map(f=>`<div>${f.change_z>=0?'↑':'↓'} ${f.label} <span class="${cls(f.change_z)}">${f.change_z.toFixed(2)}σ</span></div>`).join('')}</div>`).join('')||'<p class="muted">区间内没有显著换仓点</p>';
 metrics.innerHTML=['Stage1','SOXL','SOXX'].map(k=>{let x=m[k];return `<tr><td>${k}</td><td class="${cls(x.return_pct)}">${pct(x.return_pct)}</td><td>${pct(x.annualized_return_pct)}</td><td class="neg">${pct(x.max_drawdown_pct)}</td><td>${pct(x.volatility_pct)}</td><td>${x.sharpe.toFixed(2)}</td></tr>`}).join('');
} addEventListener('resize',()=>loadData());loadData();
</script></body></html>"""


def create_stage1_app(root: Path | str | None = None) -> FastAPI:
    source = resolve_stage1_dashboard_source(root)
    app = FastAPI(title="Stage1 Qlib Dashboard")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return APP_HTML

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ready", "source": "qlib_recorder", "recorder_id": source.recorder_id}

    @app.get("/api/stage1")
    def stage1(
        start: str | None = Query(default=None),
        end: str | None = Query(default=None),
    ) -> dict:
        try:
            return build_stage1_dashboard_payload(root=source.root, start=start, end=end)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage1 qlib recorder dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_stage1_app(args.root), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
