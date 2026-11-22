"use strict";
const el = id => document.getElementById(id);
let trace, analysis, sourceName, generation=0;
const ms = us => Number.isFinite(us) ? (us/1000).toLocaleString("en-US",{maximumFractionDigits:3})+" ms" : "Off";
function range(values, n) {
  const pct=x=>(100*x/n).toFixed(1)+"%";
  return values[0]===values[1]?pct(values[0]):pct(values[0])+"–"+pct(values[1]);
}
function update() {
  if (!trace) return;
  try {
    const deadline=Math.round(el("deadline").valueAsNumber*1000);
    analysis=analyze(trace,deadline,el("budget").valueAsNumber);
    const {train,test,best,step}=analysis, p=test.points[best];
    el("status").textContent=sourceName+" · "+train.n.toLocaleString()+" calibration / "+test.n.toLocaleString()+" held out · "+(trace.paired?"complete pairs":"censored observations");
    el("selected").textContent=ms(p.delay);
    el("miss").textContent=range(p.miss,test.n);
    el("launch").textContent=range(p.launch,test.n);
    el("inspect").max=test.points.length-1; el("inspect").value=best;
    el("explain").textContent="The first 70% of requests select a delay on a "+step+" µs grid. Among policies inside the calibration budget, the explorer minimizes deadline misses, then extra attempts. The displayed rates come from the final requests in file order. "+
    (trace.paired?"With backups off, "+range(test.points.at(-1).miss,test.n)+" miss the deadline.":"Shaded bands bound the possible results when the losing time is unknown. Selection uses the upper bounds.");
    inspect();
  } catch(e) { el("status").textContent=e.message; }
}
function inspect() {
  if (!analysis) return;
  const {test}=analysis, p=test.points[Number(el("inspect").value)];
  el("label").textContent=ms(p.delay);
  el("inspected").textContent="Held out · "+range(p.miss,test.n)+" missed the deadline · "+range(p.launch,test.n)+" launched a backup";
  draw();
}
function draw() {
  if (!analysis) return;
  const canvas=el("canvas"),w=canvas.clientWidth,h=canvas.clientHeight,scale=window.devicePixelRatio||1;
  canvas.width=w*scale;canvas.height=h*scale;
  const ctx=canvas.getContext("2d");ctx.scale(scale,scale);
  const {points,n}=analysis.test, last=points.length-2;
  const x=i=>46+(w-56)*i/Math.max(last,1), y=v=>h-23-(h-41)*v/n;
  ctx.font="11px monospace";ctx.fillStyle="#aebdc2";ctx.strokeStyle="#354044";ctx.lineWidth=1;
  for(let pct=0;pct<=100;pct+=25){const yy=y(n*pct/100);ctx.fillText(pct+"%",3,yy+3);ctx.beginPath();ctx.moveTo(46,yy);ctx.lineTo(w-8,yy);ctx.stroke();}
  for(const [key,color] of [["launch","#67c3d9"],["miss","#ff9764"]]) {
    ctx.beginPath();for(let i=0;i<=last;i++){const px=x(i),v=y(points[i][key][1]); if(i===0)ctx.moveTo(px,v);else ctx.lineTo(px,v);}
    for(let i=last;i>=0;i--)ctx.lineTo(x(i),y(points[i][key][0]));ctx.closePath();ctx.fillStyle=color+"33";ctx.fill();
    for(const side of [0,1]){ctx.beginPath();for(let i=0;i<=last;i++){if(i===0)ctx.moveTo(x(i),y(points[i][key][side]));else ctx.lineTo(x(i),y(points[i][key][side]));}ctx.strokeStyle=color;ctx.lineWidth=2;ctx.stroke();}
  }
  const i=Number(el("inspect").value);
  if(i<=last){ctx.strokeStyle="#e6ecee";ctx.lineWidth=1;ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(x(i),12);ctx.lineTo(x(i),h-19);ctx.stroke();ctx.setLineDash([]);}
  ctx.fillStyle="#aebdc2";ctx.fillText("0",46,h-4);const right=ms(points[last].delay);ctx.fillText(right,w-9-ctx.measureText(right).width,h-4);
}
async function load(fromFile) {
  const seq=++generation;
  try {
    let contents;
    if(fromFile) contents=await fromFile.text();
    else {const r=await fetch("loopback.csv");if(!r.ok)throw Error("Could not load example.");contents=await r.text();}
    if(seq!==generation)return;
    const next=parseTrace(contents);trace=next;sourceName=fromFile?fromFile.name:"Measured localhost example";update();
  } catch(e){if(seq===generation)el("status").textContent="Could not open trace. "+e.message+" Previous results are shown.";}
}
el("deadline").addEventListener("input",update);el("budget").addEventListener("input",update);
el("file").addEventListener("change",e=>{if(e.target.files[0])load(e.target.files[0]);});
el("reset").addEventListener("click",()=>{el("file").value="";el("deadline").value=6;el("budget").value=25;load(null);});
el("inspect").addEventListener("input",inspect);window.addEventListener("resize",draw);load(null);
