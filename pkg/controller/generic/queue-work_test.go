/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package generic

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/client-go/util/workqueue"
)

const queueWorkTestTimeout = 5 * time.Second

// controlledWorkQueue preserves the real queue's behavior while allowing tests
// to pause at the boundaries of Get and Done.
type controlledWorkQueue struct {
	workqueue.TypedRateLimitingInterface[int]
	beforeGet   func()
	afterGet    func(int, bool)
	beforeDone  func(int)
	doneCalls   atomic.Int32
	forgetCalls atomic.Int32
}

func (queue *controlledWorkQueue) Get() (int, bool) {
	if queue.beforeGet != nil {
		queue.beforeGet()
	}
	item, shutdown := queue.TypedRateLimitingInterface.Get()
	if queue.afterGet != nil {
		queue.afterGet(item, shutdown)
	}
	return item, shutdown
}

func (queue *controlledWorkQueue) Done(item int) {
	if queue.beforeDone != nil {
		queue.beforeDone(item)
	}
	queue.TypedRateLimitingInterface.Done(item)
	queue.doneCalls.Add(1)
}

func (queue *controlledWorkQueue) Forget(item int) {
	queue.TypedRateLimitingInterface.Forget(item)
	queue.forgetCalls.Add(1)
}

// queueWorkTest releases every test-controlled wait before shutting down the
// queue and joining the goroutines, including when an assertion times out.
type queueWorkTest struct {
	ctx         context.Context
	queue       workqueue.TypedRateLimitingInterface[int]
	releases    []func()
	completions []<-chan struct{}
}

func newQueueWorkTest(t *testing.T) *queueWorkTest {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	test := &queueWorkTest{ctx: ctx}
	t.Cleanup(func() {
		for _, release := range test.releases {
			release()
		}
		cancel()
		if test.queue != nil {
			test.queue.ShutDown()
		}
		cleanupCtx, cleanupCancel := context.WithTimeout(context.Background(), queueWorkTestTimeout)
		defer cleanupCancel()
		for i, done := range test.completions {
			select {
			case <-done:
			case <-cleanupCtx.Done():
				t.Errorf("timed out joining queue worker test goroutine %d during cleanup", i)
				return
			}
		}
	})
	return test
}

func (test *queueWorkTest) gate() (<-chan struct{}, func()) {
	gate := make(chan struct{})
	release := sync.OnceFunc(func() { close(gate) })
	test.releases = append(test.releases, release)
	return gate, release
}

func (test *queueWorkTest) run(f func()) <-chan struct{} {
	done := make(chan struct{})
	test.completions = append(test.completions, done)
	go func() {
		defer close(done)
		f()
	}()
	return done
}

func waitForQueueWork(t *testing.T, signal <-chan struct{}, description string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(queueWorkTestTimeout):
		t.Fatalf("timed out waiting for %s", description)
	}
}

func TestStopTestWorkersWaitsForProcessingAndCleanup(t *testing.T) {
	for _, blockedAt := range []string{"Process", "Done"} {
		t.Run(blockedAt, func(t *testing.T) {
			test := newQueueWorkTest(t)
			gate, release := test.gate()
			blocked := make(chan struct{})
			var processCalls atomic.Int32
			ctl := NewQueueAndWorkers(t.Name(), 1, func(context.Context, int) (error, bool) {
				processCalls.Add(1)
				if blockedAt == "Process" {
					close(blocked)
					<-gate
				}
				return nil, false
			})
			queue := &controlledWorkQueue{TypedRateLimitingInterface: ctl.Queue}
			if blockedAt == "Done" {
				queue.beforeDone = func(int) {
					close(blocked)
					<-gate
				}
			}
			ctl.Queue = queue
			test.queue = queue
			queue.Add(1)
			var keepRunning bool
			workerDone := test.run(func() { keepRunning = ctl.processNextWorkItem(test.ctx) })
			waitForQueueWork(t, blocked, blockedAt+" to block")

			const callers = 4
			entered := make(chan struct{}, callers)
			returned := make(chan struct{}, callers)
			stopsDone := make([]<-chan struct{}, 0, callers)
			for range callers {
				stopsDone = append(stopsDone, test.run(func() {
					entered <- struct{}{}
					ctl.StopTestWorkers(test.ctx)
					returned <- struct{}{}
				}))
			}
			for range callers {
				waitForQueueWork(t, entered, "StopTestWorkers caller to start")
			}
			// The channels control the worker's position; this bounded window
			// gives all callers an opportunity to reveal an early return.
			select {
			case <-returned:
				t.Errorf("StopTestWorkers returned while %s was blocked", blockedAt)
			case <-time.After(50 * time.Millisecond):
			}
			release()
			for _, done := range stopsDone {
				waitForQueueWork(t, done, "StopTestWorkers to return")
			}
			status := ctl.GetStatus()
			if status.SyncsStarted != 1 || status.SyncsFinished != 1 {
				t.Errorf("status after StopTestWorkers = %+v, want one completed sync", status)
			}
			if got := queue.doneCalls.Load(); got != 1 {
				t.Errorf("Done calls after StopTestWorkers = %d, want 1", got)
			}
			if got := queue.forgetCalls.Load(); got != 1 {
				t.Errorf("Forget calls after StopTestWorkers = %d, want 1", got)
			}
			waitForQueueWork(t, workerDone, "worker to return")
			if !keepRunning || processCalls.Load() != 1 {
				t.Errorf("worker result = %t, Process calls = %d, want true and 1", keepRunning, processCalls.Load())
			}
		})
	}
}

func TestStopTestWorkersBeforeRetrievedItemIsProcessed(t *testing.T) {
	test := newQueueWorkTest(t)
	gate, release := test.gate()
	retrieved := make(chan struct{})
	var processCalls atomic.Int32
	ctl := NewQueueAndWorkers(t.Name(), 1, func(context.Context, int) (error, bool) {
		processCalls.Add(1)
		return nil, false
	})
	queue := &controlledWorkQueue{TypedRateLimitingInterface: ctl.Queue}
	queue.afterGet = func(int, bool) {
		close(retrieved)
		<-gate
	}
	ctl.Queue = queue
	test.queue = queue
	queue.Add(1)
	var keepRunning bool
	workerDone := test.run(func() { keepRunning = ctl.processNextWorkItem(test.ctx) })
	waitForQueueWork(t, retrieved, "queue to retrieve the item")
	stopDone := test.run(func() { ctl.StopTestWorkers(test.ctx) })
	waitForQueueWork(t, stopDone, "StopTestWorkers before Get returns")
	release()
	waitForQueueWork(t, workerDone, "worker to reject the retrieved item")
	if keepRunning || processCalls.Load() != 0 {
		t.Errorf("worker result = %t, Process calls = %d, want false and 0", keepRunning, processCalls.Load())
	}
	if got := queue.doneCalls.Load(); got != 1 {
		t.Errorf("Done calls = %d, want 1", got)
	}
	status := ctl.GetStatus()
	if status.SyncsStarted != 0 || status.SyncsFinished != 0 {
		t.Errorf("status = %+v, want no started or finished syncs", status)
	}
}

func TestStopTestWorkersDoesNotWaitForEmptyQueue(t *testing.T) {
	test := newQueueWorkTest(t)
	getting := make(chan struct{})
	var processCalls atomic.Int32
	ctl := NewQueueAndWorkers(t.Name(), 1, func(context.Context, int) (error, bool) {
		processCalls.Add(1)
		return nil, false
	})
	queue := &controlledWorkQueue{TypedRateLimitingInterface: ctl.Queue}
	queue.beforeGet = func() { close(getting) }
	ctl.Queue = queue
	test.queue = queue
	var keepRunning bool
	workerDone := test.run(func() { keepRunning = ctl.processNextWorkItem(test.ctx) })
	waitForQueueWork(t, getting, "worker to enter Get on an empty queue")
	stopDone := test.run(func() { ctl.StopTestWorkers(test.ctx) })
	waitForQueueWork(t, stopDone, "StopTestWorkers with Get waiting")
	queue.ShutDown()
	waitForQueueWork(t, workerDone, "worker to leave the shut down queue")
	if keepRunning || processCalls.Load() != 0 || queue.doneCalls.Load() != 0 {
		t.Errorf("worker result = %t, Process calls = %d, Done calls = %d, want false, 0, 0", keepRunning, processCalls.Load(), queue.doneCalls.Load())
	}
	status := ctl.GetStatus()
	if status.SyncsStarted != 0 || status.SyncsFinished != 0 {
		t.Errorf("status = %+v, want no started or finished syncs", status)
	}
}

func TestStopTestWorkersAfterInitialSync(t *testing.T) {
	const workers = 3
	test := newQueueWorkTest(t)
	gate, release := test.gate()
	processing := make(chan struct{}, workers)
	processedSync := make(chan struct{})
	var processCalls atomic.Int32
	ctl := NewKnowsProcessedSync(t.Name(), workers, func(context.Context, int) (error, bool) {
		processCalls.Add(1)
		processing <- struct{}{}
		<-gate
		return nil, false
	}, func(worker int) int { return -worker - 1 }, func(item int) bool { return item < 0 }, func(context.Context) {
		close(processedSync)
	})
	queue := &controlledWorkQueue{TypedRateLimitingInterface: ctl.Queue}
	workersLeavingGet := make(chan struct{})
	var shutdownGets atomic.Int32
	queue.afterGet = func(_ int, shutdown bool) {
		if shutdown && shutdownGets.Add(1) == workers {
			close(workersLeavingGet)
		}
	}
	ctl.Queue = queue
	test.queue = queue
	for item := range workers {
		queue.Add(item)
	}
	if err := ctl.StartWorkers(test.ctx); err != nil {
		t.Fatalf("StartWorkers: %v", err)
	}
	// StartWorkers owns its goroutines. On cleanup, cancel their context,
	// shut down their queue, and wait until all blocked Gets are released.
	test.completions = append(test.completions, processedSync, workersLeavingGet)
	for range workers {
		waitForQueueWork(t, processing, "all workers to process concurrently")
	}
	release()
	waitForQueueWork(t, processedSync, "initial sentinels to complete")
	if !ctl.HasProcessedSync() {
		t.Fatal("initial-sync callback ran before HasProcessedSync became true")
	}
	stopDone := test.run(func() { ctl.StopTestWorkers(test.ctx) })
	waitForQueueWork(t, stopDone, "StopTestWorkers after initial sync")
	status := ctl.GetStatus()
	if status.SyncsStarted != 2*workers || status.SyncsFinished != 2*workers {
		t.Errorf("status after StopTestWorkers = %+v, want %d completed syncs including sentinels", status, 2*workers)
	}
	if got := processCalls.Load(); got != workers {
		t.Errorf("Process calls = %d, want %d ordinary items", got, workers)
	}
	if got := queue.doneCalls.Load(); got != 2*workers {
		t.Errorf("Done calls after StopTestWorkers = %d, want %d", got, 2*workers)
	}
}
