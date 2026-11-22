#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using Micros = std::chrono::microseconds;
using Ns = std::int64_t;

Ns since(Clock::time_point epoch) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - epoch).count();
}

struct Branch { Ns submit=-1, dequeued=-1, start=-1, end=-1; };
struct Request {
    int service_us[2];
    Branch branch[2];
    Ns planned, first=-1;
};

// FIFO tickets ensure a waiting worker cannot lose a slot to a new arrival.
class Slots {
    std::mutex mutex;
    std::condition_variable changed;
    int free=2;
    std::uint64_t next=0, serving=0;
public:
    void acquire() {
        std::unique_lock lock(mutex);
        auto ticket = next++;
        changed.wait(lock, [&]{ return ticket == serving && free > 0; });
        ++serving;
        --free;
        changed.notify_all();
    }
    void release() {
        std::lock_guard lock(mutex);
        ++free;
        changed.notify_all();
    }
};

class Pool {
    int arm;
    Clock::time_point epoch;
    std::vector<Request>& requests;
    Slots& slots;
    std::mutex& response_mutex;
    std::mutex mutex;
    std::condition_variable ready;
    std::deque<size_t> work;
    std::vector<std::thread> threads;
    bool stop=false;
    void worker() {
        for (;;) {
            size_t id;
            {
                std::unique_lock lock(mutex);
                ready.wait(lock, [&]{return stop || !work.empty();});
                if (work.empty()) return;
                id=work.front();
                work.pop_front();
            }
            auto& r=requests[id];
            auto& b=r.branch[arm];
            b.dequeued=since(epoch);
            slots.acquire();
            b.start=since(epoch);
            std::this_thread::sleep_for(Micros(r.service_us[arm]));
            {
                std::lock_guard lock(response_mutex);
                b.end=since(epoch);
                if (r.first == -1) r.first=b.end;
            }
            slots.release();
        }
    }
public:
    Pool(int arm, Clock::time_point epoch, std::vector<Request>& requests, Slots& slots,
         std::mutex& response_mutex) :
        arm(arm), epoch(epoch), requests(requests), slots(slots), response_mutex(response_mutex) {
        for(int i=0;i<2;i++) threads.emplace_back([this]{worker();});
    }
    void submit(size_t id) {
        {
            std::lock_guard lock(mutex);
            requests[id].branch[arm].submit=since(epoch);
            work.push_back(id);
        }
        ready.notify_one();
    }
    void finish() {
        {
            std::lock_guard lock(mutex);
            stop=true;
        }
        ready.notify_all();
        for(auto& thread:threads) thread.join();
    }
};

int main(int argc, char** argv) {
    if(argc != 3) { std::cerr<<"load RATE DELAY_US|-1 < service_times.txt\n"; return 2; }
    const int rate=std::stoi(argv[1]), delay=std::stoi(argv[2]);
    if(rate <= 0 || delay < -1) return 2;
    std::vector<Request> requests;
    int primary=0, backup=0;
    while(std::cin>>primary>>backup) {
        if(primary<0 || backup<0) return 2;
        Request r{};
        r.service_us[0]=primary;
        r.service_us[1]=backup;
        r.planned=static_cast<Ns>(requests.size())*1000000000LL/rate;
        requests.push_back(r);
    }
    if(requests.empty()) return 2;
    const auto epoch=Clock::now()+Micros(40000);
    std::mutex response_mutex;
    Slots slots;
    Pool p(0,epoch,requests,slots,response_mutex), b(1,epoch,requests,slots,response_mutex);
    size_t arrived=0, next_timer=0;
    // Relative to actual primary submit. Timer deadlines preserve request order.
    while(arrived<requests.size() || (delay>=0 && next_timer<arrived)) {
        const bool timer_due=delay>=0 && next_timer<arrived;
        Ns due=timer_due ? requests[next_timer].branch[0].submit+Ns(delay)*1000 : 0;
        if(arrived<requests.size() && (!timer_due || requests[arrived].planned<=due)) {
            std::this_thread::sleep_until(epoch+std::chrono::nanoseconds(requests[arrived].planned));
            p.submit(arrived++);
        } else {
            std::this_thread::sleep_until(epoch+std::chrono::nanoseconds(due));
            std::lock_guard lock(response_mutex);
            if(requests[next_timer].first == -1) b.submit(next_timer);
            ++next_timer;
        }
    }
    p.finish();
    b.finish();
    std::cout<<"id,planned_ns,first_ns,primary_submit_ns,primary_dequeued_ns,primary_start_ns,primary_end_ns,"
        "backup_submit_ns,backup_dequeued_ns,backup_start_ns,backup_end_ns\n";
    for(size_t i=0;i<requests.size();i++) {
        auto& r=requests[i];
        std::cout<<i+1<<','<<r.planned<<','<<r.first;
        for(auto& arm:r.branch) std::cout<<','<<arm.submit<<','<<arm.dequeued<<','<<arm.start<<','<<arm.end;
        std::cout<<'\n';
    }
}
