"""Original animated graphics. Each one is drawn from real aggregate counts and has a still frame for reduced motion."""
import base64
import json
import math

import streamlit as st

_HEAD = """<style>html,body{margin:0;background:transparent;overflow:hidden}canvas{width:100%;display:block}
#tip{position:fixed;pointer-events:none;background:#0C1624;border:1px solid rgba(112,217,208,.45);border-radius:9px;color:#F3F5F2;font:12px 'JetBrains Mono',Consolas,monospace;padding:6px 9px;opacity:0;transition:opacity .15s;white-space:nowrap}</style>"""

_RINGS = _HEAD + """<canvas id=c role="img" aria-label="Concentric rings by age. Points are strokes in the development data. Most lie on the outer rings, above age 60."></canvas><div id=tip></div><script>
const D=__DATA__,H=__H__,still=matchMedia('(prefers-reduced-motion: reduce)').matches,c=document.getElementById('c'),g=c.getContext('2d'),tip=document.getElementById('tip');
let W,cx,cy,R,mx=-99,my=-99;const dpr=Math.min(devicePixelRatio||1,2);
function size(){W=innerWidth;c.width=W*dpr;c.height=H*dpr;c.style.height=H+'px';g.setTransform(dpr,0,0,dpr,0,0);cx=W/2;cy=H/2;R=Math.min(W,H)/2-22}
addEventListener('resize',()=>{size();if(still)draw(0)});size();
let s=7;const rnd=()=>(s=(s*16807)%2147483647)/2147483647;
const N=[];D.bands.forEach((b,i)=>{const r0=(i+.6)/(D.bands.length+.1);for(let k=0;k<b.people;k++)N.push({i,r:r0+(rnd()-.5)*.07,a:rnd()*6.283,v:(.05+rnd()*.05)*(i%2?1:-1)/(1+i*.35),k:0});
for(let k=0;k<b.strokes;k++)N.push({i,r:r0+(rnd()-.5)*.09,a:rnd()*6.283,v:(.05+rnd()*.05)*(i%2?1:-1)/(1+i*.35),k:1,p:rnd()*6.28})});
c.onmousemove=e=>{mx=e.offsetX;my=e.offsetY};c.onmouseleave=()=>{mx=-99};
function draw(t){t/=1000;g.clearRect(0,0,W,H);
D.bands.forEach((b,i)=>{const r=R*(i+.6)/(D.bands.length+.1);g.beginPath();g.arc(cx,cy,r,0,6.283);g.strokeStyle='rgba(112,217,208,'+(.07+.02*i)+')';g.lineWidth=1;g.stroke()});
const q=[];for(const n of N){const a=n.a+(still?0:t*n.v),x=cx+Math.cos(a)*n.r*R,y=cy+Math.sin(a)*n.r*R;n.x=x;n.y=y;if(n.k)q.push(n)}
g.lineWidth=.6;for(let i=0;i<q.length;i++)for(let j=i+1;j<q.length;j++){const dx=q[i].x-q[j].x,dy=q[i].y-q[j].y,d=dx*dx+dy*dy;if(d<900){g.strokeStyle='rgba(217,164,91,'+(.32*(1-d/900))+')';g.beginPath();g.moveTo(q[i].x,q[i].y);g.lineTo(q[j].x,q[j].y);g.stroke()}}
for(const n of N){g.beginPath();if(n.k){const z=2.5+(still?0:.5*Math.sin(t*1.4+n.p));g.arc(n.x,n.y,z,0,6.283);g.fillStyle='#D9A45B'}else{g.arc(n.x,n.y,1.15,0,6.283);g.fillStyle='rgba(112,217,208,.5)'}g.fill()}
const gl=g.createRadialGradient(cx,cy,0,cx,cy,46);gl.addColorStop(0,'rgba(37,194,177,.95)');gl.addColorStop(.12,'rgba(37,194,177,.5)');gl.addColorStop(1,'rgba(37,194,177,0)');g.fillStyle=gl;g.beginPath();g.arc(cx,cy,46,0,6.283);g.fill();g.fillStyle='#F3F5F2';g.beginPath();g.arc(cx,cy,3.2,0,6.283);g.fill();
g.font="500 11px 'JetBrains Mono',Consolas,monospace";g.textAlign='center';D.bands.forEach((b,i)=>{const y=cy-R*(i+.6)/(D.bands.length+.1);const w=g.measureText(b.label).width+10;g.fillStyle='#101C2C';g.fillRect(cx-w/2,y-8,w,16);g.fillStyle='#C6D2DC';g.fillText(b.label,cx,y+4)});
if(mx>0){const d=Math.hypot(mx-cx,my-cy)/R*(D.bands.length+.1)-.6,i=Math.round(d);if(i>=0&&i<D.bands.length&&Math.abs(d-i)<.5){const b=D.bands[i];tip.textContent='age '+b.label+': '+b.strokes+' strokes of '+b.total.toLocaleString('en')+' people';tip.style.opacity=1;tip.style.left=Math.min(mx+14,W-260)+'px';tip.style.top=(my+14)+'px'}else tip.style.opacity=0}else tip.style.opacity=0;
if(!still)setTimeout(()=>requestAnimationFrame(draw),30)}
draw(0);if(still)c.onmousemove=e=>{mx=e.offsetX;my=e.offsetY;draw(0)};
</script>"""


def age_rings(people, strokes, height=500):
    """people/strokes: counts for 5-year bins from age 0. Teal points: one per 12 people. Amber points: one per stroke."""
    bands = []
    for label, a, b in [("0–29", 0, 6), ("30s", 6, 8), ("40s", 8, 10), ("50s", 10, 12), ("60s", 12, 14), ("70+", 14, 18)]:
        n, s = sum(people[a:b]), sum(strokes[a:b])
        bands.append(dict(label=label, total=n, strokes=s, people=round((n - s) / 12)))
    st.iframe(_RINGS.replace("__DATA__", json.dumps(dict(bands=bands))).replace("__H__", str(height)), height=height)


_LENS = _HEAD + """<canvas id=c role="img" aria-label="Two fields of 170 records. Among records with no BMI, 37 had a stroke. Among the same number of records with BMI, about 7."></canvas><script>
const D=__DATA__,H=__H__,still=matchMedia('(prefers-reduced-motion: reduce)').matches,c=document.getElementById('c'),g=c.getContext('2d'),dpr=Math.min(devicePixelRatio||1,2);let W,P=[],mx=null,my;
let s=11;const rnd=()=>(s=(s*16807)%2147483647)/2147483647;
const mark=(n,k)=>{const a=Array(n).fill(0).map((_,i)=>i<k);for(let i=n-1;i>0;i--){const j=rnd()*(i+1)|0;[a[i],a[j]]=[a[j],a[i]]}return a};
const A=mark(D.n,D.miss),B=mark(D.n,D.have);
function size(){W=innerWidth;c.width=W*dpr;c.height=H*dpr;c.style.height=H+'px';g.setTransform(dpr,0,0,dpr,0,0);P=[];const gap=34,fw=(W-gap)/2,cols=13,st=Math.min(fw/cols,(H-56)/14);
[A,B].forEach((F,f)=>F.forEach((k,i)=>P.push({x:f*(fw+gap)+st*(i%cols+.5),y:44+st*((i/cols|0)+.5),k,f,r:st*.3})))}
addEventListener('resize',()=>{size();if(still)draw(0)});size();c.onmousemove=e=>{mx=e.offsetX;my=e.offsetY};c.onmouseleave=()=>mx=null;
function draw(t){t/=1000;g.clearRect(0,0,W,H);const fw=(W-34)/2;g.textAlign='left';g.font="500 12.5px 'Inter','Segoe UI',sans-serif";g.fillStyle='#C6D2DC';g.fillText('BMI not recorded',2,18);g.fillText(fw<200?'BMI recorded':'BMI recorded, same-size sample',fw+36,18);
const L=64,lx=mx??W*(.5+.42*Math.sin(t*.42)),ly=mx!==null&&mx!==undefined?my:44+(H-70)*(.5+.32*Math.sin(t*.73));
for(const p of P){const d=Math.hypot(p.x-lx,p.y-ly),on=still||d<L,k=p.k&&(on||true),z=on?1.28:1;g.beginPath();
if(p.k){g.arc(p.x,p.y,p.r*z,0,6.283);g.fillStyle=on?'#D9A45B':'rgba(217,164,91,.42)';g.fill()}
else if(p.f==0){g.arc(p.x,p.y,p.r*.82*z,0,6.283);g.strokeStyle=on?'rgba(112,217,208,.95)':'rgba(112,217,208,.32)';g.lineWidth=1.2;g.setLineDash([2.4,2.2]);g.stroke();g.setLineDash([])}
else{g.arc(p.x,p.y,p.r*.7*z,0,6.283);g.fillStyle=on?'rgba(112,217,208,.8)':'rgba(112,217,208,.24)';g.fill()}}
if(!still){g.beginPath();g.arc(lx,ly,L,0,6.283);g.strokeStyle='rgba(243,245,242,.75)';g.lineWidth=2;g.stroke();const gl=g.createRadialGradient(lx,ly,L*.2,lx,ly,L);gl.addColorStop(0,'rgba(112,217,208,.10)');gl.addColorStop(1,'rgba(112,217,208,.02)');g.fillStyle=gl;g.fill();g.beginPath();g.moveTo(lx+L*.71,ly+L*.71);g.lineTo(lx+L*1.06,ly+L*1.06);g.lineWidth=5;g.lineCap='round';g.strokeStyle='rgba(243,245,242,.6)';g.stroke();setTimeout(()=>requestAnimationFrame(draw),30)}}
draw(0);</script>"""


def lens(n, miss, have, height=300):
    st.iframe(_LENS.replace("__DATA__", json.dumps(dict(n=n, miss=miss, have=have))).replace("__H__", str(height)), height=height)


def svg_img(svg, alt="", style=""):
    """st.html strips inline SVG, so vector graphics are embedded as an image."""
    return f'<img src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}" alt="{alt}" style="{style}">'


def gauge(risk, prev, threshold, top=.5, r=120):
    """270-degree dial from 0 to 50%. The highest estimate the model can give for allowed inputs is below 50%."""
    full = 2 * math.pi * r * .75
    pt = lambda v, rad: (160 + rad * math.cos(math.radians(135 + 270 * v / top)), 160 + rad * math.sin(math.radians(135 + 270 * v / top)))
    tick = "".join(f'<text x="{pt(v, r + 24)[0]:.1f}" y="{pt(v, r + 24)[1] + 4:.1f}" text-anchor="middle" font-size="11" fill="#A3B4C4" font-family="Consolas,monospace">{v:.0%}</text>' for v in (0, .1, .2, .3, .4, .5))
    (a, b), (c, d) = pt(threshold, r - 13), pt(threshold, r + 13)
    track = f'<circle cx="160" cy="160" r="{r}" fill="none" stroke-width="16" stroke-linecap="round" transform="rotate(135 160 160)" '
    val = f"{risk:.1%}" if risk is not None else "–"
    to, fr = (full * min(v or 0, top) / top for v in (risk, prev))
    arc = "" if risk is None else track + f'class="arc" stroke="url(#gg)" stroke-dasharray="{to:.1f} 1000"/>'
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 282">'
           f'<style>.arc{{animation:a .8s cubic-bezier(.2,.8,.2,1)}}@keyframes a{{from{{stroke-dasharray:{fr:.1f} 1000}}}}@media(prefers-reduced-motion:reduce){{.arc{{animation:none}}}}</style>'
           '<defs><linearGradient id="gg" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#25C2B1"/><stop offset="1" stop-color="#70D9D0"/></linearGradient></defs>'
           + track + f'stroke="#A3B4C4" stroke-opacity=".16" stroke-dasharray="{full:.1f} 1000"/>' + arc + tick
           + f'<line x1="{a:.1f}" y1="{b:.1f}" x2="{c:.1f}" y2="{d:.1f}" stroke="#D9A45B" stroke-width="3.5"/></svg>')
    label = f"Estimated probability of the stroke class {val}, on a dial from 0 to 50 percent. Research threshold {threshold:.1%}."
    return f'<div class="gauge">{svg_img(svg, label)}<div class="gv"><b>{val}</b><span>stroke-class probability</span></div><small>amber mark: research threshold {threshold:.1%}</small></div>'
