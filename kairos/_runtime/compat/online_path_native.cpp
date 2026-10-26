// Complete residual calendar evaluator. Same restricted model as the Python
// reference; native acceleration changes neither candidates nor objective.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

struct Row {
    int stage=-1, computed=0, generated=0, pages=0;
    double ready=0.; bool started=false, pending=true, done=false;
};

extern "C" int online_calendar(int n, const int* offsets, const double* turns,
    const double* initial, const int* minimum, const int* gates,
    const double* cfg, double* out, double* events, int event_capacity) {
    const int capacity=int(cfg[0])-int(cfg[1]), page=int(cfg[2]);
    const int maxseq=int(cfg[3]), budget=int(cfg[4]);
    const bool newpeak=cfg[5]!=0.;
    const double floor=cfg[6], host=cfg[7];
    std::vector<Row> rows(n); std::vector<int> peaks(n), completed(n,0);
    std::vector<double> finish(n,0.);
    auto turn = [&](int p,int k,int field){return turns[4*(offsets[p]+k)+field];};
    auto callpages = [&](int p,int k){return int(std::ceil((turn(p,k,0)+turn(p,k,1))/page));};
    auto target = [&](int p,int k){return std::max(callpages(p,k),k==0?minimum[p]:0);};
    int used=0, active=0, done=0, count=0, steps=0, queries=0, peak_used=0;
    double t=0., area=0., reserved_area=0., engine=0.;
    auto event = [&](int kind,int p,int stage){
        if(count>=event_capacity) return false;
        double* e=events+5*count++;e[0]=kind;e[1]=p;e[2]=stage;e[3]=t;e[4]=used;
        return true;
    };
    auto open = [&](int p,int k,double ready,int held){
        Row& r=rows[p];r.stage=k;r.ready=ready;r.generated=0;r.pages=held;r.started=false;
        r.computed=k?std::min(int(turn(p,k,2))/page*page,std::max(0,int(turn(p,k,0))-1)):0;
    };
    for(int p=0;p<n;++p) {
        peaks[p]=minimum[p];
        for(int k=0;k<offsets[p+1]-offsets[p];++k)peaks[p]=std::max(peaks[p],callpages(p,k));
        if(peaks[p]>capacity){out[10]=3;return 0;}
        const double* s=initial+6*p;
        if(s[0]) {
            open(p,0,s[1],int(s[4]));rows[p].computed=int(s[2]);rows[p].generated=int(s[3]);
            rows[p].started=s[5]!=0.;rows[p].pending=false;used+=rows[p].pages;++active;
            if(rows[p].generated>=turn(p,0,1)){out[10]=4;return 0;}
        }
    }
    if(used>capacity){out[10]=2;return 0;}
    while(done<n) {
        if(steps>1000000){out[10]=5;return 0;}
        for(int p=0;p<n;++p) {
            Row& r=rows[p];if(r.stage<0||r.done||r.ready>t||r.started)continue;
            int want=target(p,r.stage);if(newpeak)want=std::max(want,r.pages);
            if(used-r.pages+want<=capacity){used+=want-r.pages;r.pages=want;r.started=true;
                if(!event(1,p,r.stage)){out[10]=6;return 0;}}
        }
        while(active<maxseq) {
            int p=-1;for(int q=0;q<n;++q)if(rows[q].pending){p=q;break;}
            if(p<0)break;
            bool gate=true;for(int q=0;q<n;++q)if(completed[q]<gates[p*n+q]){gate=false;break;}
            int want=newpeak?peaks[p]:target(p,0);
            if(!gate||used+want>capacity)break;
            rows[p].pending=false;open(p,0,t,want);rows[p].started=true;used+=want;++active;
            if(!event(0,p,0)){out[10]=6;return 0;}
        }
        peak_used=std::max(peak_used,used);
        std::vector<int> decoders,prefills,members;
        for(int p=0;p<n;++p){const Row& r=rows[p];
            if(r.stage>=0&&!r.done&&r.started&&r.ready<=t)
                (r.computed>=turn(p,r.stage,0)?decoders:prefills).push_back(p);}
        if(decoders.empty()&&prefills.empty()) {
            double future=std::numeric_limits<double>::infinity();
            for(const Row& r:rows)if(r.stage>=0&&!r.done&&r.ready>t)future=std::min(future,r.ready);
            if(!std::isfinite(future)){out[10]=1;return 0;}
            area+=(n-done)*(future-t);reserved_area+=used*(future-t);t=future;continue;
        }
        int left=budget,Q=0,prefill=0;double A=0.;
        decoders.insert(decoders.end(),prefills.begin(),prefills.end());
        for(int p:decoders){if(!left)break;Row& r=rows[p];int prompt=int(turn(p,r.stage,0)),amount;
            if(r.computed<prompt){amount=std::min(left,prompt-r.computed);int begin=r.computed+1;
                A+=double(amount)*(2.*begin+amount-1)/2;r.computed+=amount;prefill+=amount;
                if(r.computed==prompt)++r.generated;
            }else{amount=1;A+=prompt+r.generated;++r.generated;}
            Q+=amount;left-=amount;members.push_back(p);
        }
        int N=int(members.size());if(!N){out[10]=5;return 0;}
        double ds=cfg[12+N];
        double duration=(prefill?std::max({floor,cfg[8]+cfg[9]*Q+cfg[10]*A+cfg[11]*N,ds}):ds)+host;
        int repeat=1;
        if(!prefill){int until=1000000000;double future=std::numeric_limits<double>::infinity();
            for(int p:members)until=std::min(until,int(turn(p,rows[p].stage,1))-rows[p].generated);
            for(const Row& r:rows)if(r.stage>=0&&!r.done&&r.ready>t)future=std::min(future,r.ready);
            int event_steps=std::isfinite(future)?std::max(1,int(std::ceil((future-t)/duration-1e-10))):until+1;
            repeat=std::min(until+1,event_steps);
            for(int p:members)rows[p].generated+=repeat-1;
        }
        if(duration<=0||repeat<=0){out[10]=5;return 0;}
        double dt=duration*repeat;area+=(n-done)*dt;reserved_area+=used*dt;t+=dt;
        engine+=dt;queries+=prefill;steps+=repeat;
        for(int p:members){Row& r=rows[p];int k=r.stage;
            if(r.generated<turn(p,k,1))continue;
            completed[p]=k+1;if(!event(2,p,k)){out[10]=6;return 0;}
            if(k+1==offsets[p+1]-offsets[p]){finish[p]=t;r.done=true;used-=r.pages;--active;++done;
                if(!event(3,p,k)){out[10]=6;return 0;}
            }else{int held=callpages(p,k);if(newpeak)held=std::max(held,r.pages);
                used+=held-r.pages;double ready=t+turn(p,k,3);open(p,k+1,ready,held);}
        }
    }
    double total=0.;for(double c:finish)total+=c;
    if(std::abs(total-area)>1e-5*std::max(1.,total)){out[10]=7;return 0;}
    out[0]=1.;out[1]=total/n;out[2]=total;out[3]=t;out[4]=queries;out[5]=engine;
    out[6]=steps;out[7]=peak_used;out[8]=reserved_area;out[9]=count;
    return 1;
}
