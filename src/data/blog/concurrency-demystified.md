---
title: "Concurrency, Demystified: What Every Engineer Should Actually Know"
pubDatetime: 2026-08-23T00:00:00Z
modDatetime: 2026-08-23T00:00:00Z
slug: concurrency-demystified
featured: false
draft: false
tags:
  - interview-prep
  - systems-design
  - concurrency
  - python
description: "A from-fundamentals tour of concurrency in Python, told as one connected story instead of a glossary of terms — threads vs. processes, the GIL, locks, condition variables, asyncio, distributed rate limiting, and a staged concurrent web crawler exercise."
---

Concurrency has a reputation problem. Ask most engineers what they remember about it and you'll hear "locks," "race conditions," and a vague sense of dread left over from a data structures class. But strip away the jargon and it's really just one question, asked over and over in different contexts: **what happens when more than one thing tries to touch the same piece of state at the same time?**

This post walks through the core ideas — from threads and the GIL, up through locks, async, rate limiting, and the classic problems that show up in interviews — as one connected story rather than a glossary of terms.

---

## Threads, processes, and the difference that actually matters

A **process** is an isolated running program with its own memory. A **thread** is a unit of execution *inside* a process, sharing that process's memory with every other thread in it.

That shared memory is the whole story. It's why threads are cheap to spin up and easy to coordinate — no serialization, no IPC — and it's also the exact reason concurrency bugs exist at all. Processes don't have this problem, because they don't share anything by default.

The practical split:

- **CPU-bound work** (image processing, number crunching) → processes, because you actually need multiple cores running simultaneously.
- **I/O-bound work** (network calls, disk reads, waiting on a database) → threads or async, because the bottleneck is *waiting*, not computing, and you don't need separate cores to wait faster.

```python
import threading
import multiprocessing

def do_work():
    ...

# Thread — cheap, shares memory with the parent
t = threading.Thread(target=do_work)
t.start()
t.join()

# Process — separate memory space, real parallel execution
p = multiprocessing.Process(target=do_work)
p.start()
p.join()
```

## Wait — does a process even run anything by itself?

Not quite, and this trips people up. A **thread is the actual unit of execution** — it's what the OS scheduler runs. A **process is a container**: it owns the memory space and other resources, but it doesn't execute code by itself. Every process has *at least one thread* (the "main thread"), and that's what actually runs your code. So `multiprocessing.Process(target=do_work)` doesn't just create an inert process — it creates a process *and* a main thread inside it that runs `do_work`. Nothing stops you from spawning additional `threading.Thread`s inside that process either, which gives you a process (isolation) containing several threads (shared memory with each other, not with anything outside).

It's also worth being precise about *when* new processes or threads get created at all, since it's easy to picture them appearing all over the place. They don't. When you run `python my_script.py`, the OS creates exactly **one process** for the entire program — start to finish. Every function call, every variable, every loop runs inside that single process, on its single main thread, one instruction at a time. Calling a function doesn't spawn a process or a thread; it just pushes a new frame onto the same thread's call stack and returns when it's done.

```python
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

result = add(2, 3)        # still the one process, one thread
result2 = multiply(4, 5)  # still the same one — nothing spawned here
```

New processes or threads only appear when you explicitly ask for one:

```python
from multiprocessing import Process

def do_work():
    ...

p = Process(target=do_work)  # THIS is what creates a second process
p.start()
```

Only at `p.start()` does the OS create a second, independent process — with its own interpreter and memory space — to run `do_work`. Your original script keeps running in the first process; now there are two, running independently. The same logic applies to threads: `threading.Thread(...)` is what creates one, not ordinary function calls.

The mental model worth keeping: **one process per running program, by default.** More processes or threads only exist because your code explicitly created them — via `multiprocessing.Process`, `subprocess.run`/`Popen`, `os.fork()`, or a thread/process pool. Function calls, variable assignments, loops, conditionals — none of that touches processes or threads; it's all just instructions running sequentially within whatever process and thread are already executing.

## The GIL: Python's asterisk on everything

If you're writing Python, there's a wrinkle: the **Global Interpreter Lock**. CPython only lets one thread execute Python bytecode at a time — full stop, regardless of how many cores you have.

This sounds like it defeats the purpose of threading, and for CPU-bound work, it basically does. Ten threads doing matrix math in Python will not run any faster than one, because they're taking turns on the same lock. But here's the nuance that separates a shallow answer from a good one: a thread **releases the GIL while it's blocked on I/O**. So if your threads are mostly waiting — on a socket, a file, a database — the GIL barely matters, because the threads aren't competing for CPU time, they're taking turns waiting.

That's the whole reason "threading helps with I/O-bound concurrency but not CPU-bound parallelism" is such a common line in interviews — it's not a trivia fact, it's the load-bearing insight for when to reach for threads, processes, or async at all.

You can see the effect directly. This CPU-bound function gets no benefit from threads:

```python
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def cpu_bound(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

N = 20_000_000

start = time.time()
[cpu_bound(N) for _ in range(4)]
print("sequential:", time.time() - start)

start = time.time()
with ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(cpu_bound, [N] * 4))
print("threads:", time.time() - start)  # about the same as sequential — GIL contention

start = time.time()
with ProcessPoolExecutor(max_workers=4) as ex:
    list(ex.map(cpu_bound, [N] * 4))
print("processes:", time.time() - start)  # noticeably faster — real parallelism
```

Run this yourself and the threaded version won't beat the sequential one — sometimes it's even slightly slower, from the overhead of threads handing the GIL back and forth. The process version is where you'll actually see the speedup.

## What actually happens inside that ProcessPoolExecutor call

It's worth being precise about what a worker process actually has access to, because it's easy to picture it as "running your whole script" — it isn't.

**What gets sent to a worker is just the function and its arguments**, pickled and sent over a pipe. The worker unpickles them, calls the function, pickles the return value, and sends *that* back over another pipe. The worker doesn't have access to your other variables, open files, or anything else living in the parent's memory — it's a separate process with separate memory.

But how does the worker know what `cpu_bound` even *is*, code-wise? That depends on the OS:

- **On Linux** (the default `fork` start method): the child is created as a copy of the parent's entire memory at that instant — it already has your module loaded, `cpu_bound` included. No re-import needed.
- **On Windows, and macOS by default since Python 3.8** (the `spawn` start method): there's no `fork()` available, so Python starts a fresh interpreter process and has it **re-import your script as a module** to get the function definition. The top level of your file genuinely re-runs in the child.

That's exactly why this guard shows up everywhere:

```python
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex:
        list(ex.map(cpu_bound, [N] * 4))
```

On `spawn` systems, skip the guard and the child re-imports the file, hits `ProcessPoolExecutor(...)` again at the top level, and spawns its *own* pool of children — recursive process creation. The guard works because `__name__` is `"__main__"` only for the original invocation; every re-import by a child sees `__name__` as the module's actual name instead, and skips the block.

**What about a global variable that `cpu_bound` mutates?** No race condition — but not for a subtle reason, a structural one: processes don't share memory, so there's nothing to race over.

```python
counter = 0

def cpu_bound(n):
    global counter
    counter += 1   # mutates THIS process's own copy, not the parent's
    return n

with ProcessPoolExecutor(max_workers=4) as ex:
    list(ex.map(cpu_bound, [10, 20, 30, 40]))

print(counter)  # still 0 in the parent — completely unaffected
```

Each worker has its own independent copy of `counter`. Incrementing it only changes that worker's copy; nothing syncs back to the parent except whatever the function explicitly `return`s. Four workers "racing" on a global doesn't corrupt anything, because they're never touching the same memory. This is the flip side of the isolation tradeoff from earlier: with threads, a shared global is a real hazard needing a lock; with processes, that same global is automatically safe *because it isn't actually shared* — though also not useful for coordinating state across processes. If you genuinely need processes to share and update a counter together, a plain global won't do it — you need `multiprocessing.Value` or `multiprocessing.Manager`, which set up real shared memory and, notably, come with their own internal locking, because as soon as memory actually is shared across processes, the race condition problem is back. One more wrinkle: under `fork`, workers get a copy-on-write snapshot of the global as it stood at fork time; under `spawn`, since the module gets freshly re-imported, workers get the global's original initialization value instead — which can differ if the parent mutated it before creating the pool.

**Does the parent wait for the workers, and how do results come back?** By default, yes, it waits — and the mechanism depends on the API:

```python
with ProcessPoolExecutor(max_workers=4) as ex:
    results = ex.map(cpu_bound, [N] * 4)   # returns immediately — lazy iterator
    results = list(results)                 # THIS blocks until all are done
```

`ex.map()` itself returns instantly — an iterator, not a list. The blocking happens when you iterate it, and it blocks **in submission order**: it waits for task 0's result first, then task 1's, even if task 3 actually finishes first. Even without `list()`, exiting the `with` block calls `.shutdown(wait=True)` internally, which blocks until every submitted task is done.

For more control, `submit()` gives you a `Future` per task instead:

```python
futures = [ex.submit(cpu_bound, N) for _ in range(4)]

for f in futures:
    print(f.result())   # blocks until THIS specific future is done
```

`.result()` is what blocks here, and you can call it in submission order, or use `as_completed(futures)` to process results in the order they actually finish — useful when task durations vary a lot.

Physically, each worker pickles its return value and writes it to a pipe back to the parent; a background thread inside the executor reads from that pipe, unpickles the value, and matches it to the right `Future` by an internal task ID. It's message-passing, not shared memory — each result travels back individually and gets slotted into place by the executor's own bookkeeping, never by a race-prone shared variable.

## Managing threads without babysitting them

Manually creating and destroying threads for every unit of work is wasteful. A **thread pool** fixes this: a fixed set of worker threads pull tasks off a shared queue, run them, and go back to waiting for the next one.

```python
from concurrent.futures import ThreadPoolExecutor

def fetch(url):
    ...  # some I/O-bound call

with ThreadPoolExecutor(max_workers=5) as executor:
    results = executor.map(fetch, urls)
```

Under the hood: workers spin up to `max_workers`, idle on an internal queue, wake up when work arrives, and shut down cleanly once the pool's context exits and all submitted work is done. `ProcessPoolExecutor` is the identical API swapped over to processes — the interface doesn't change, only what's parallelizing does.

## Where things actually go wrong: race conditions

Here's the canonical failure:

```python
counter = 0

def increment():
    global counter
    for _ in range(100_000):
        counter += 1
```

Run this from two threads and you'd expect `200_000`. You'll usually get less. `counter += 1` isn't one operation — it's read, add, write — and if two threads interleave those three steps, one thread's update can quietly overwrite the other's. This is a **race condition**: correctness depends on timing you don't control.

## Locks: the blunt but reliable fix

A **lock** (or mutex) guarantees only one thread executes a given block of code at a time. Wrap the shared state:

```python
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100_000):
        with lock:
            counter += 1
```

Now it's reliably `200_000`. The protected block is called the **critical section** — the minimum span of code that touches shared state and therefore needs exclusivity. Good concurrent code keeps critical sections as small as possible; the more code you hold a lock over, the more you're serializing threads that could otherwise run freely.

## The cost of locks: deadlock

Locks solve races but introduce a new failure mode. If thread A holds lock 1 and waits on lock 2, while thread B holds lock 2 and waits on lock 1 — neither ever proceeds. That's a **deadlock**.

Here it is in code — two threads, each grabbing the same two locks in opposite order:

```python
import threading

lock_a = threading.Lock()
lock_b = threading.Lock()

def worker_1():
    with lock_a:
        with lock_b:   # waits for lock_b, held by worker_2
            print("worker_1 done")

def worker_2():
    with lock_b:
        with lock_a:   # waits for lock_a, held by worker_1
            print("worker_2 done")

threading.Thread(target=worker_1).start()
threading.Thread(target=worker_2).start()
# This can hang forever — neither print statement may ever run.
```

The fix is almost always structural: acquire locks in a **consistent global order** everywhere in your codebase — e.g. always `lock_a` before `lock_b`, in both functions — keep the number of locks held at once as small as possible, and consider timeouts on acquisition so a thread gives up rather than waiting indefinitely:

```python
if lock_a.acquire(timeout=2):
    try:
        if lock_b.acquire(timeout=2):
            try:
                ...
            finally:
                lock_b.release()
    finally:
        lock_a.release()
```

## Semaphores: locks with a headcount

A lock allows exactly one thread in. A **semaphore** allows up to *N* — useful when the constraint isn't "exclusive access" but "bounded concurrency," like capping how many simultaneous requests hit a rate-limited API:

```python
semaphore = threading.Semaphore(3)

def call_api():
    with semaphore:
        ...  # at most 3 threads here at once
```

A semaphore with a limit of 1 behaves like a lock, but with one meaningful difference: a lock is owned by whoever acquired it and only they can release it, while a semaphore can be released by any thread. That distinction is what makes semaphores useful for signaling between threads, not just mutual exclusion.

## Condition variables: waiting without spinning

Sometimes a thread doesn't need exclusivity — it needs to **wait until something becomes true**, without burning CPU checking in a loop. Without one, you'd end up **busy-waiting**:

```python
while not queue:
    pass  # spins, checking millions of times a second, for nothing
item = queue.popleft()
```

This "works," but it pins a CPU core at 100% the entire time it's waiting. A **condition variable** fixes this: it lets a thread go properly to sleep — zero CPU — and only wake up when something might have changed.

```python
condition = threading.Condition()

def consumer():
    with condition:
        while not queue:
            condition.wait()   # sleeps, releases the lock while asleep
        item = queue.popleft()

def producer():
    with condition:
        queue.append(1)
        condition.notify()   # wakes one sleeping thread
```

`wait()` does two things atomically: it releases the lock (so someone else can actually get in and change something) and puts the thread to sleep at the OS level. When `notify()` fires, the sleeping thread wakes, automatically re-acquires the lock, and resumes right where it left off — back at the `while` check.

That `while`, not `if`, matters: waking up doesn't guarantee the condition is still true. Some other thread might grab the lock first and change things again before you get to run. So you always re-verify in a loop rather than trusting the wakeup itself.

### `notify()` vs. `notify_all()`, with more than two threads

None of this is limited to one producer and one consumer — it generalizes to any number of threads. The part that changes is *who* gets woken.

`notify()` wakes exactly **one** arbitrary sleeping thread — not necessarily the one that's been waiting longest, and the API makes no ordering guarantee. With several consumers asleep, one item added wakes exactly one of them; the rest stay asleep until the next `notify()`. This is efficient — you don't want to wake five consumers for one item, since four would just wake up, find nothing, and go back to sleep for nothing.

```python
def consumer(name):
    while True:
        with condition:
            while not queue:
                condition.wait()
            item = queue.popleft()
        print(f"{name} got {item}")

def producer():
    for i in range(10):
        with condition:
            queue.append(i)
            condition.notify()   # wakes ONE sleeping consumer
```

This works fine with multiple consumers and a single producer. It gets riskier once you add **multiple producers that also wait** — say, a bounded buffer where producers block when it's full. Now two different groups are sleeping on the same condition: consumers waiting for "item available," producers waiting for "space available." A plain `notify()` has no idea which group to wake — it might wake another sleeping producer instead of the consumer that should actually run, and everyone stalls.

That's why the bounded-buffer version typically uses `notify_all()` — wake everyone, and let each thread's own `while` loop decide whether it's actually their turn:

```python
condition = threading.Condition()
buffer = collections.deque()
MAX_SIZE = 5

def producer(name):
    for i in range(10):
        with condition:
            while len(buffer) >= MAX_SIZE:
                condition.wait()          # wait for space
            buffer.append(i)
            condition.notify_all()        # wake everyone — let them recheck

def consumer(name):
    while True:
        with condition:
            while not buffer:
                condition.wait()          # wait for an item
            item = buffer.popleft()
            condition.notify_all()        # wake everyone — let them recheck
```

It's mildly wasteful — some woken threads immediately find their own condition still false and go straight back to sleep — but it's correct, which bare `notify()` isn't once two different groups are waiting for two different things on the same condition object.

**Rule of thumb:** one kind of waiter, one kind of event → `notify()`. Multiple different groups waiting for different things on the same condition → `notify_all()`, unless you've very carefully reasoned through exactly who could be asleep.

### There's no built-in notion of "producer" or "consumer"

Worth being explicit about this: `wait()` and `notify()` are completely generic. The `Condition` object doesn't know or care what role is calling it — `notify()` can't say "wake a consumer specifically." It just wakes some sleeping thread, whoever that happens to be.

If you want signals to only ever reach the intended role, the cleaner fix is **separate condition variables per role**, sharing the same underlying lock (since they're protecting the same data):

```python
buffer = collections.deque()
MAX_SIZE = 5
lock = threading.Lock()
not_full = threading.Condition(lock)   # producers wait here
not_empty = threading.Condition(lock)  # consumers wait here

def producer(name):
    for i in range(10):
        with not_full:
            while len(buffer) >= MAX_SIZE:
                not_full.wait()
            buffer.append(i)
        with not_empty:
            not_empty.notify()   # can only ever wake a consumer

def consumer(name):
    while True:
        with not_empty:
            while not buffer:
                not_empty.wait()
            item = buffer.popleft()
        with not_full:
            not_full.notify()    # can only ever wake a producer
```

Each condition has its own separate "waiting room," so a signal can only land where you intend — you get the efficiency of `notify()` (wake exactly one) with the correctness `notify_all()` was buying you on a single shared condition.

### Doesn't `notify_all()` waking everyone cause a race?

It looks like it should, but no — because **waking up and acquiring the lock are two separate steps**, and the lock is what serializes everything.

`notify_all()` only marks every sleeping thread as "eligible to run again." It does not hand the lock to all of them at once — that would defeat the purpose of having a lock in the first place. What actually happens:

1. `notify_all()` fires; every sleeping thread becomes runnable, but the lock is still held by whoever called it.
2. That thread exits its `with` block and releases the lock.
3. All the woken threads now compete for the lock — but a lock only ever grants it to one thread at a time. One wins; the rest just block on the lock, same as ordinary contention.
4. The winner resumes right after its `wait()` call, rechecks its `while` condition while holding the lock exclusively, then either proceeds or calls `wait()` again — releasing the lock as it does.
5. The next thread in line for the lock gets it, does its own check, and so on — one at a time until everyone's had their turn.

So it looks like a simultaneous wakeup, but every check-and-maybe-modify still happens in strict isolation, funneled one at a time through the lock — the exact same guarantee a lock always provides. Concretely: if one item is added and three consumers plus a producer were sleeping, one consumer gets the lock, finds an item, takes it, and releases. The next consumer gets the lock, rechecks, finds the buffer empty again, and goes straight back to sleep — a small amount of wasted work, but never two threads inside the critical section at once. That's the "slightly wasteful" cost of `notify_all()` mentioned above — never an unsafe one.

This pattern — one or more threads producing, one or more consuming, coordinated safely — is the backbone of one of the most common designs in concurrent systems:

## Producer-consumer

One or more threads produce work; one or more threads consume it; a thread-safe queue sits in between. Python's `queue.Queue` implements the locking and condition-variable logic internally, so you rarely write it by hand:

```python
q = queue.Queue(maxsize=10)  # bounded

def producer():
    q.put(item)   # blocks if full

def consumer():
    item = q.get()  # blocks if empty
```

The `maxsize` isn't incidental — it's **backpressure**. Without a bound, a fast producer and a slow consumer means unbounded memory growth. A bounded queue forces the producer to slow down to match the consumer instead.

## "Thread-safe" is doing more work than it sounds like

A structure is thread-safe when its internal operations are already protected, so callers don't need their own locking. But it's worth being precise about *which* operations. Java's plain `HashMap` isn't thread-safe at all; `Hashtable` is, but locks the entire table for every operation; `ConcurrentHashMap` is thread-safe *and* scales, because it only locks the segment being modified rather than the whole structure. That distinction — safety versus safety-with-throughput — is usually the actual point of the question when someone asks you to compare them.

## A quick vocabulary check: busy-waiting, blocking, and non-blocking

These three terms get used loosely and conflated a lot, so it's worth pinning them down precisely before going further — especially since async is where the distinction actually matters.

**Busy-waiting (a.k.a. spinning)** is a thread repeatedly checking a condition in a tight loop, doing nothing else, burning CPU the whole time:

```python
while not ready:
    pass  # checking, checking, checking — full CPU, zero useful work
```

This is almost always bad in ordinary code, but it shows up intentionally in low-level systems code as a **spinlock**, where the expected wait is so short (a few CPU cycles) that actually sleeping and waking the thread would cost more than just spinning briefly.

**Blocking** is a call that does not return control to your code until it's done — your thread stops at that line and waits:

```python
data = socket.recv(1024)  # thread pauses here until data arrives
print("only runs after recv returns")
```

Blocking is *not* the same as busy-waiting. A blocking call typically puts the thread to sleep at the OS level — zero CPU used — and something else wakes it back up. `lock.acquire()`, `condition.wait()`, `queue.get()`, `socket.recv()` — all blocking, all efficient (no spinning), but all stop your thread dead until they return.

**Non-blocking** is a call that returns immediately regardless of whether the work is done, usually telling you "not ready yet" instead of waiting:

```python
sock.setblocking(False)
try:
    data = sock.recv(1024)
except BlockingIOError:
    data = None  # nothing available right now — but we didn't wait for it
```

| Concept | Waits? | Uses CPU while waiting? |
|---|---|---|
| Busy-waiting | yes | yes (spinning) |
| Blocking call | yes | no (sleeping) |
| Non-blocking call | no | no (returns immediately) |

`condition.wait()` from the previous section is a good example that ties this together: it's a *blocking* call — your thread genuinely stops there — that's specifically implemented to *avoid* busy-waiting, which was the whole motivation for condition variables in the first place.

### Who actually does the waking?

This is the part that's easy to gloss over: for `condition.wait()`, the wakeup is something *you* built — another thread in your program calls `notify()`. If nothing ever calls it, you sleep forever. That's purely application-level coordination.

`socket.recv()` is different — there's no other thread involved by default. What actually happens: your thread calls `recv()`, a system call that hands off to the kernel. If data's already sitting in the socket's buffer, the kernel returns it immediately. If not, the kernel marks your thread "blocked on this socket" and takes it off the CPU entirely — the scheduler simply doesn't run it. Separately, when a network packet arrives, the network card raises a hardware interrupt; the kernel's driver appends the data to the socket's buffer, marks your thread runnable again, and the scheduler eventually resumes it right where `recv()` left off.

So "who wakes the thread" splits into two categories:

- **Application-level blocking** (`condition.wait()`, `lock.acquire()`) — you're waiting on something another thread *in your own program* will do, and it's your own code (`notify()`, `release()`) that triggers the wakeup. (Lock contention is usually implemented internally using the same kind of OS wait-queue mechanism as a condition variable.)
- **OS/hardware-level blocking** (`socket.recv()`, `f.read()`, `time.sleep()`) — you're waiting on something outside your program entirely — network, disk, a hardware clock — and the kernel tracks the wait and performs the wakeup, triggered by a hardware interrupt or timer, not by any code you wrote.

Both genuinely block the thread and use no CPU while waiting. What differs is *who's* responsible for eventually waking you back up.

### Synchronous vs. asynchronous — a different axis entirely

This one's about program *structure*, not about any single call:

- **Synchronous** code executes one step after another, in order, each step completing before the next starts.
- **Asynchronous** code kicks work off and continues doing other things, getting notified — or checking back — when that work finishes.

### Two more failure modes worth distinguishing from deadlock

- **Starvation**: a thread *could* eventually run, but keeps getting passed over — e.g. a scheduler consistently favors other threads, so this one waits indefinitely even though nothing is actually stuck. Different from deadlock, where progress is *impossible*, not just unlucky.
- **Livelock**: threads aren't stuck — they're actively doing something — but that something never leads to progress. The classic image: two people in a hallway both stepping the same direction to let the other pass, repeatedly, forever.

## Async: concurrency without threads at all

Threads rely on the OS switching between them. **asyncio** does something structurally different: a single thread runs an event loop, and your code voluntarily yields control at every `await`.

```python
async def fetch(session, url):
    async with session.get(url) as response:
        return await response.text()

async def main():
    async with aiohttp.ClientSession() as session:
        return await asyncio.gather(*(fetch(session, u) for u in urls))
```

Calling asyncio "non-blocking" is close but worth being precise about, because it still blocks *somewhere* — just not where you'd expect. Under the hood, the event loop waits for I/O using genuinely non-blocking OS facilities like `epoll`/`select`/`kqueue` — a single thread can ask the kernel "tell me the moment any of these 500 sockets are ready, and let me do other things until then." That part really is non-blocking.

But **your code** still has each coroutine *logically* pause at every `await` — that coroutine's own execution stops there and doesn't continue until the awaited thing resolves. The difference from threading isn't "blocking vs. non-blocking" so much as **what blocks**: with threading, the whole OS thread blocks, and the OS scheduler picks what runs next. With asyncio, only that one coroutine's progress pauses, while the underlying thread stays completely free — the event loop immediately runs a *different* coroutine on that same thread in the meantime. The thread itself is never idle; only individual pieces of your async code are, one at a time, in sequence.

So the more accurate framing: asyncio uses a non-blocking mechanism internally so the *thread* never blocks, while still giving you code that *reads* like ordinary blocking, sequential logic. That's why it's sometimes described as "blocking-style code with non-blocking-style performance" — you write it top to bottom like blocking code, but the OS thread underneath is never stuck the way a real blocking call would leave it.

One thread can juggle thousands of open connections this way, because it's never actually blocked — it's just hopping to whatever task is ready to make progress. Nothing preempts you mid-statement, which is why async code often needs *fewer* locks than threaded code.

The catch, and a favorite follow-up question in interviews: this non-blocking property only holds as long as *everything* in the call chain is genuinely `await`-based and cooperative. Call a real blocking function inside an async function — plain `requests.get()`, or `time.sleep()` instead of `asyncio.sleep()` — and that call blocks the underlying OS thread for real, freezing the entire event loop and every other coroutine waiting on it. Async isn't a free upgrade over threading; it's a different tradeoff, better suited to very high connection counts and worse suited to anything with a blocking call buried in it.

## Rate limiting: concurrency meeting the outside world

Once your system talks to something with finite capacity — a database, an inference backend, a third-party API — concurrency and rate limiting become the same conversation.

**Token bucket** is the standard pattern: a bucket holds up to *N* tokens, refilled at a fixed rate. Every request consumes a token; an empty bucket means the request waits or gets rejected. It allows bursts up to the bucket size while still enforcing an average rate over time.

**Sliding window** counts requests within a trailing time window rather than a fixed one, avoiding the "boundary burst" problem of fixed windows (where a client can send double its limit by timing requests around the window edge) at the cost of tracking more state.

A minimal token bucket, thread-safe within a single process:

```python
import time
import threading

class TokenBucket:
    def __init__(self, capacity, refill_rate):
        self.capacity = capacity
        self.tokens = capacity
        self.refill_rate = refill_rate  # tokens added per second
        self.last_refill = time.time()
        self.lock = threading.Lock()

    def allow(self):
        with self.lock:
            now = time.time()
            self.tokens = min(
                self.capacity,
                self.tokens + (now - self.last_refill) * self.refill_rate,
            )
            self.last_refill = now
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

bucket = TokenBucket(capacity=5, refill_rate=1)  # burst of 5, refills 1/sec
for i in range(8):
    print(f"request {i}:", "allowed" if bucket.allow() else "rejected")
```

That version works fine for a single process. In a real deployment behind a load balancer, though, you rarely have one process — you have a fleet of servers, and the limit needs to apply to *all of them combined*. That's where things get genuinely harder.

### Why in-memory rate limiting falls apart across machines

If each server keeps its own `TokenBucket` in local memory, a client isn't actually limited to *N* requests — they're limited to *N* requests **per server**. Ten servers behind a round-robin load balancer means a client can effectively get 10x the intended limit, just by having their requests land on different machines. The lock in the code above protects a bucket from threads *within one process* — it does nothing for buckets living in ten separate processes that don't even know about each other.

The fix is the same one from earlier in the post: **move the shared state out of any single process's memory and into something all instances can see** — typically Redis, since it's fast and supports atomic operations. But this reintroduces the exact race condition from the very first example in this post, just relocated from threads inside a process to *requests hitting different servers*:

```python
# NOT safe — read-then-write across a network is not atomic
current = redis_client.get("count:user123")
if int(current or 0) < LIMIT:
    redis_client.incr("count:user123")
    allow_request()
```

Two servers can both read `current` as `4` (limit 5) at nearly the same instant, both decide "still under the limit," and both increment — count ends up `6`, over the limit. Same read-modify-write race as `counter += 1` on threads, just with network latency between the steps instead of instruction interleaving.

### Making the check atomic

Redis executes a single command atomically, but a check-then-increment is *two* commands, with a gap between them where another server can sneak in. The fix is the same principle as wrapping `counter += 1` in a `threading.Lock()` from earlier in this post — make the read-modify-write indivisible, so no other party can observe or act on a half-updated state. In practice this usually means reaching for whatever atomic primitive the shared store provides — a single combined command instead of separate read-then-write calls, a conditional write ("set this value, but only if it still matches what I last read"), or a short server-side script that the store runs as one uninterruptible unit. The mechanism differs by store; the goal is always the same one this post keeps coming back to: no gap between "read" and "write" for anyone else to land in.

### Sliding window, distributed

The sliding-window-log approach — keep a timestamp per request, count how many fall within the trailing window, evict the rest — needs the same atomicity treatment: "drop old entries, count what's left, and record this request" has to happen as one unit, or two servers can both see room under the limit and both let a request through that pushes the count over. It's more precise than a fixed window (no boundary-burst problem) but costs more memory, since you're storing a timestamp per request within the window rather than a single counter — and that cost scales with request volume.

### The tradeoffs distributed rate limiting adds on top

- **A shared store is a new dependency and a new bottleneck.** Every request now makes a round trip to the shared store before it can proceed — network latency gets added to your request path, and that store itself becomes something that needs its own capacity planning and failure handling.
- **What happens when the shared store is unreachable?** This is a real design decision, not an edge case to shrug off — **fail open** (let requests through, prioritizing availability) versus **fail closed** (reject requests, prioritizing protecting the backend). Which one's correct depends entirely on which failure mode is more expensive for your system.
- **Clock skew matters.** Both the token bucket and sliding window approaches above depend on timestamps. If different servers' clocks disagree even slightly, "now" means slightly different things depending on which server computed it — usually solved by sourcing time from the shared store itself (most support fetching their own server-side clock) rather than trusting each application server's local clock.
- **A single shared limiter reintroduces a single point of contention** — exactly the tradeoff locks always carry, just at datacenter scale instead of thread scale. A common middle ground is **local + global limiting**: give each server a small local token bucket as a fast first-pass filter (cheap, no network round trip, catches obviously-over-limit traffic instantly) backed by a shared check against the store for the actual enforced limit — trading a bit of precision for a lot less load on the shared store.

The throughline, again: whether it's a `+= 1` on a shared integer between two threads, or a `GET`-then-`SET` on a shared key between two servers, it's the same race condition at a different scale — and the fix is always some version of "make the read-modify-write atomic," whether that's a `threading.Lock`, a conditional write, or a database transaction.

## The two problems everyone eventually reinvents

**Dining Philosophers**: *N* philosophers share *N* forks between them, each needing two adjacent forks to eat. If everyone grabs their left fork simultaneously, everyone waits forever for their right — deadlock, structurally identical to the lock-ordering problem above. Fix it the same way: impose an ordering (pick up the lower-numbered fork first) or cap how many philosophers can attempt to eat at once.

**Reader-Writer**: many readers can safely access shared data at once; a writer needs exclusive access. The standard solution tracks an active-reader count alongside a write lock — the first reader blocks writers, the last reader releases that block, and readers in between just increment and decrement a counter.

Both problems are old, but they're really just race conditions and deadlocks wearing a costume — which is exactly why they're worth knowing by shape, not by memorized solution.

Dining Philosophers, with the fix applied — each philosopher picks up the lower-numbered fork first, which breaks the circular wait:

```python
import threading

NUM_PHILOSOPHERS = 5
forks = [threading.Lock() for _ in range(NUM_PHILOSOPHERS)]

def philosopher(i):
    left, right = i, (i + 1) % NUM_PHILOSOPHERS
    first, second = min(left, right), max(left, right)  # consistent order

    with forks[first]:
        with forks[second]:
            print(f"philosopher {i} is eating")

for i in range(NUM_PHILOSOPHERS):
    threading.Thread(target=philosopher, args=(i,)).start()
```

Reader-Writer, with a reader count gating a single write lock:

```python
import threading

class ReadWriteLock:
    def __init__(self):
        self.read_count = 0
        self.read_count_lock = threading.Lock()
        self.write_lock = threading.Lock()

    def acquire_read(self):
        with self.read_count_lock:
            self.read_count += 1
            if self.read_count == 1:
                self.write_lock.acquire()  # first reader blocks writers

    def release_read(self):
        with self.read_count_lock:
            self.read_count -= 1
            if self.read_count == 0:
                self.write_lock.release()  # last reader unblocks writers

    def acquire_write(self):
        self.write_lock.acquire()

    def release_write(self):
        self.write_lock.release()
```

## When things fail anyway

No amount of correct locking prevents failure — networks drop, nodes crash, downstream services slow down. The concurrency-adjacent patterns for handling this:

- **Timeouts** — never wait indefinitely on a lock, a call, or another thread.
- **Retries with exponential backoff and jitter** — retry transient failures, but with increasing delay and a little randomness, so a fleet of clients doesn't retry in lockstep and hammer a recovering service all at once.
- **Circuit breakers** — after repeated failures, stop calling a struggling downstream service for a while instead of piling on more retries.
- **Graceful shutdown** — let in-flight work finish or cancel cleanly, rather than killing threads mid-operation and leaving shared state half-updated.

Retry with exponential backoff and jitter:

```python
import random
import time

def call_with_retry(fn, max_attempts=5, base_delay=1):
    for attempt in range(max_attempts):
        try:
            return fn()
        except TransientError:
            if attempt == max_attempts - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)  # backoff + jitter
            time.sleep(delay)
```

A bare-bones circuit breaker — after enough consecutive failures, stop calling the downstream service for a cooldown period:

```python
import time

class CircuitBreaker:
    def __init__(self, failure_threshold=3, cooldown=30):
        self.failure_threshold = failure_threshold
        self.cooldown = cooldown
        self.failures = 0
        self.opened_at = None

    def call(self, fn):
        if self.opened_at and time.time() - self.opened_at < self.cooldown:
            raise Exception("circuit open — failing fast")

        try:
            result = fn()
            self.failures = 0
            self.opened_at = None
            return result
        except Exception:
            self.failures += 1
            if self.failures >= self.failure_threshold:
                self.opened_at = time.time()
            raise
```

Notice what's missing: a lock. `self.failures += 1` is the exact same read-modify-write hazard as `counter += 1` from the very first example in this post — fine if `call()` is only ever invoked from one thread at a time, but a real race if multiple threads share one `CircuitBreaker` instance. Everything this post has argued about protecting shared state applies to the post's own example code.

## Seeing it all in one system

It's worth noticing how many of these ideas show up together in something as ordinary as a **concurrent web crawler**: a bounded thread pool doing the crawling, a thread-safe queue acting as the frontier of URLs to visit, a locked set to avoid re-crawling the same page, a semaphore capping requests per host, timeouts and backoff for flaky pages, and a clean shutdown once the frontier empties out.

None of these ideas are exotic on their own. What makes concurrency hard isn't any single concept — it's that real systems need several of them at once, each covering a failure mode the others don't.

### Exercise: build the crawler yourself

Here's the problem laid out the way it tends to actually get asked — a small base spec, then a series of constraints added one at a time. Try implementing each stage before reading the next; each one forces you to reuse a different concept from this post.

**Stage 1 — Sequential baseline.**
Given a starting URL, crawl up to `max_pages` pages. For each page, extract its links and follow them, breadth-first, until you hit the limit or run out of new links. Don't visit the same URL twice. Get this working single-threaded first — it's the correctness baseline everything else gets checked against.

*What this stage is really testing:* whether you reach for concurrency by default or only once it's actually needed. A sequential, correct version first is usually the right instinct — parallelizing broken logic just gets you wrong answers faster.

**Stage 2 — Parallelize it.**
Now crawl with multiple workers at once, using a bounded thread pool (`max_workers` workers, not one thread per page). You'll need:
- A **thread-safe queue** as the frontier — the set of URLs discovered but not yet crawled.
- A **thread-safe visited set** — checking "have I seen this URL?" and adding it if not has to be atomic, or two workers can both see a URL as unvisited and crawl it twice.

*The trap here:* `if url not in visited: visited.add(url)` is a classic check-then-act race — exactly the shape of the `counter += 1` race earlier in this post, just on a set instead of an integer. It needs a lock around both operations together, not just around the `.add()`.

**Stage 3 — Don't hammer any single host.**
Add a constraint: no more than `k` concurrent requests to the same domain, even though the overall pool has many more workers than that. This is a direct application of a **semaphore** — but not one semaphore, *one per host*, created lazily as new domains are discovered.

*What this stage is testing:* whether you reach for a semaphore instead of just shrinking `max_workers` globally, which would needlessly throttle requests to *other* hosts that aren't the bottleneck.

**Stage 4 — Handle flaky pages.**
Some requests will time out or fail outright. Add a timeout per request, and retry transient failures with exponential backoff and jitter, up to a max attempt count, before giving up on that URL and moving on.

*What this stage is testing:* the fault-tolerance patterns from earlier — specifically that a single slow or dead host shouldn't be able to stall the whole crawl (that's what the per-request timeout prevents) or cause a thundering herd of synchronized retries (that's what jitter prevents).

**Stage 5 — Bounded memory.**
Assume the site graph is huge — the frontier could grow to millions of URLs if left unbounded. Cap the frontier queue's size, so a burst of newly discovered links doesn't blow up memory; workers finding new links should block (briefly) if the frontier is full, rather than the queue growing without limit.

*What this stage is testing:* recognizing backpressure — the same reason `queue.Queue(maxsize=...)` was bounded in the producer-consumer section — and applying it somewhere a first pass usually leaves unbounded.

**Stage 6 — Graceful shutdown.**
Support a stop signal — the crawl should stop accepting new work immediately, let any in-flight requests finish (or time out on their own), and return cleanly, rather than leaving threads hanging or partially-written state behind.

*What this stage is testing:* whether shutdown is treated as a first-class part of the design or an afterthought — in a real system, "how does this stop safely" usually matters as much as "how does this start."

**A question worth sitting with before you code anything:** at each stage, what's the actual critical section — the minimum amount of code that needs to run under a lock — and what can safely happen *outside* any lock? (Fetching a page over the network, for instance, should never happen while holding the visited-set's lock — that would serialize your entire crawl behind network latency, defeating the whole point of parallelizing it.) That question, more than any specific primitive, is the one that separates a working concurrent design from a merely correct-looking one.

---

## Appendix: best practices for writing the parent process

A practical checklist for `ProcessPoolExecutor`-style code, pulling together everything above.

**1. Always guard the entry point**
```python
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=4) as ex:
        ...
```
On `spawn` systems (Windows, macOS default) this isn't optional — skip it and you get runaway recursive process creation, since the child re-imports the module and hits the same pool-creation code again.

**2. Always use a context manager (`with`)**
```python
with ProcessPoolExecutor(max_workers=4) as ex:
    ...
```
This guarantees `.shutdown(wait=True)` runs even if an exception is thrown partway through — no orphaned worker processes left hanging around after a crash.

**3. Handle exceptions from futures explicitly**
An exception raised inside a worker doesn't crash the parent — it gets pickled and stored on the `Future`, silently, until you call `.result()`:
```python
futures = [ex.submit(risky_fn, x) for x in items]
for f in futures:
    try:
        print(f.result())
    except Exception as e:
        print(f"task failed: {e}")
```
`ex.map()` surfaces an exception when you iterate — but it stops iteration right there, so one bad input can silently cut off all the results after it. `submit()` plus a loop over individual futures is usually safer when failures are possible.

**4. Size `max_workers` to the actual resource, not a guess**
- CPU-bound → roughly `os.cpu_count()`, since more than that just adds context-switch overhead with no extra parallelism.
- I/O-bound (threads) → can go much higher than core count, since threads are mostly waiting, not competing for CPU.

**5. Only pass and return picklable data**
Arguments and return values cross a process boundary by being pickled. Open file handles, database connections, lambdas, and locks generally can't be pickled — pass the *inputs needed to create* a connection, not the connection itself, and let each worker open its own.

**6. Don't rely on a plain global for cross-process state**
Each process gets its own copy of a global, as covered earlier. If workers genuinely need to share and mutate state, use `multiprocessing.Value`, `Manager`, or a `Queue` — and know that as soon as you do, you're back to needing locks, because real sharing brings real races.

**7. Prefer `as_completed` when task durations vary**
```python
from concurrent.futures import as_completed
for f in as_completed(futures):
    print(f.result())
```
`map()`, or looping over `futures` in submission order, makes you wait on a slow task even if faster ones already finished. `as_completed` processes results as they actually land.

**8. Set timeouts on `.result()` for anything that can hang**
```python
try:
    result = f.result(timeout=10)
except TimeoutError:
    ...
```
Otherwise a stuck worker blocks the parent indefinitely — the same "never wait forever" principle from the fault-tolerance section above.

**9. Batch small tasks instead of submitting thousands individually**
Each `submit()` has pickling and IPC overhead. For a huge number of tiny tasks, chunk them (`map(fn, items, chunksize=100)`) so each worker processes a batch per round-trip instead of one item per round-trip — otherwise the overhead can dwarf the actual work.

**10. Keep the parent process itself lean**
The parent's main thread does real work too — pickling arguments, unpickling results, bookkeeping futures. Avoid loading large, unnecessary data structures into the parent's memory, since unlike workers (which come and go), the parent lives for the whole program's duration.
