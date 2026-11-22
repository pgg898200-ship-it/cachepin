/* Browser replay uses integer microseconds. Infinity represents an unknown upper bound. */
"use strict";
function parseTrace(text) {
  const lines = text.trim().replace(/^\uFEFF/, "").split(/\r?\n/);
  const header = lines.shift().trim();
  const paired = header === "primary_us,backup_us";
  const rich = header === "a_status,a_us,b_status,b_us";
  if (!paired && !rich && header !== "winner,elapsed_us,old_delay_us")
    throw Error("See Trace format for paired, winner, and observation CSV headers.");
  function duration(s, line) {
    if (!/^\d+$/.test(s) || !Number.isSafeInteger(Number(s)) || Number(s) > 2 ** 48)
      throw Error("Invalid duration on line " + line + ". Use integer microseconds.");
    return Number(s);
  }
  const rows = lines.map((line, i) => {
    const fields = line.trim().split(",").map(x => x.trim());
    if (fields.length !== (paired ? 2 : rich ? 4 : 3)) throw Error("Wrong field count on line " + (i+2));
    if (paired) {
      const [p,b] = fields.map(s => duration(s, i+2));
      return [p,b,p,b];
    }
    if (rich) {
      const obs=(status,value) => {
        if (status==="unstarted" && value==="") return [0,Infinity];
        if (status==="interval") {
          const edges=value.split("..").map(v=>duration(v,i+2));
          if(edges.length===2 && edges[0]<=edges[1])return edges;
        }
        const v=duration(value,i+2);
        if(status==="exact")return [v,v];
        if(status==="at_least")return [v,Infinity];
        if(status==="censored")return [v+1,Infinity];
        throw Error("Invalid status on line "+(i+2));
      };
      const [pl,ph]=obs(fields[0],fields[1]), [bl,bh]=obs(fields[2],fields[3]);
      return [pl,bl,ph,bh];
    }
    const t = duration(fields[1], i+2), d = duration(fields[2], i+2);
    if (fields[0] === "primary") return [t,Math.max(0,t-d),t,Infinity];
    if (fields[0] === "backup" && t >= d) return [t+1,t-d,Infinity,t-d];
    throw Error("Invalid winner or launch time on line " + (i+2));
  });
  if (rows.length < 2) throw Error("Add at least two requests for chronological validation.");
  return {paired, rows};
}
function endpointCurve(rows, start, end, side, deadline, step) {
  const k = Math.floor(deadline/step) + 1;
  const launches = new Float64Array(k+1), misses = new Float64Array(k+1);
  let primaryMisses = 0;
  for (let i=start; i<end; ++i) {
    const p = rows[i][side], b = rows[i][side+1];
    ++launches[0]; --launches[Math.min(k,Math.ceil(p/step))];
    if (p > deadline) {
      ++primaryMisses;
      const first = Math.max(0, Math.floor((deadline-b)/step)+1);
      if (first < k) ++misses[first];
    }
  }
  for (let i=1; i<k; ++i) { launches[i]+=launches[i-1]; misses[i]+=misses[i-1]; }
  return {launches,misses,primaryMisses};
}
function analyze(trace, deadline, budget) {
  if (!Number.isSafeInteger(deadline) || deadline < 1 || deadline > 2**40 ||
      !Number.isFinite(budget) || budget<0 || budget>100) throw Error("Check deadline and budget.");
  const step = Math.max(1,Math.ceil(deadline/16384));
  const cut = Math.min(trace.rows.length-1, Math.max(1, Math.floor(trace.rows.length*.7)));
  function evaluate(start,end) {
    const lo=endpointCurve(trace.rows,start,end,0,deadline,step);
    const hi=trace.paired?lo:endpointCurve(trace.rows,start,end,2,deadline,step);
    const points=Array.from({length:Math.floor(deadline/step)+1}, (_,i) => ({
      delay:i*step, miss:[lo.misses[i],hi.misses[i]],launch:[lo.launches[i],hi.launches[i]]}));
    points.push({delay:Infinity,miss:[lo.primaryMisses,hi.primaryMisses],launch:[0,0]});
    return {points,n:end-start};
  }
  const train=evaluate(0,cut), test=evaluate(cut,trace.rows.length);
  let best=train.points.length-1;
  train.points.forEach((p,i) => {
    const b=train.points[best];
    if (p.launch[1]*100 <= budget*train.n &&
        (p.miss[1]<b.miss[1] || (p.miss[1]===b.miss[1] && (p.launch[1]<b.launch[1] ||
         (p.launch[1]===b.launch[1] && p.delay<b.delay))))) best=i;
  });
  return {train,test,best,step};
}
if (typeof module !== "undefined") module.exports={parseTrace,analyze};
