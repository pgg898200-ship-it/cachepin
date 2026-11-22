const assert=require("node:assert/strict");
const {parseTrace,analyze}=require("../web/curve.js");
for(const text of [
  "primary_us,backup_us\n0,0\n7,2\n2,7\n9,8\n4,0\n8,9",
  "winner,elapsed_us,old_delay_us\nbackup,5,2\nprimary,4,4\nprimary,5,4\nbackup,3,3",
  "a_status,a_us,b_status,b_us\nexact,3,unstarted,\nat_least,8,interval,2..4\nunstarted,,censored,1\nexact,2,exact,3"
]){
  const t=parseTrace(text);
  for(let deadline=1;deadline<=19;deadline++)for(let budget=0;budget<=100;budget+=10){
    const {train,test,best}=analyze(t,deadline,budget);
    for(const [split,rows] of [[train,t.rows.slice(0,train.n)],[test,t.rows.slice(train.n)]]){
      for(const point of split.points)for(let end=0;end<2;end++){
        const k=2*end,d=point.delay;
        const expected=rows.filter(row=>Math.min(row[k],d+row[k+1])>deadline).length;
        assert.equal(point.miss[end],expected);
        assert.equal(point.launch[end],rows.filter(row=>row[k]>d).length);
      }
    }
    assert.ok(train.points[best].launch[1]*100<=budget*train.n);
  }
}
console.log("Browser compiler matches direct replay on paired and censored endpoints.");
