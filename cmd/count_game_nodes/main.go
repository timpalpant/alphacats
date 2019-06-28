// Script to estimate the number of nodes touched in an external sampling run.
package main

import (
	"expvar"
	"flag"
	"net/http"
	_ "net/http/pprof"
	"runtime"
	"sync"

	"github.com/golang/glog"
	"github.com/timpalpant/go-cfr"
	"github.com/timpalpant/go-cfr/tree"

	"github.com/timpalpant/alphacats"
	"github.com/timpalpant/alphacats/cards"
)

var workInProgress = expvar.NewInt("work_in_progress")

func main() {
	flag.Parse()

	go http.ListenAndServe("localhost:4124", nil)

	workCh := make(chan countJob, runtime.NumCPU())
	for i := 0; i < cap(workCh); i++ {
		go func() {
			for job := range workCh {
				doJob(job, workCh)
			}
		}()
	}
	defer close(workCh)

	deck := cards.CoreDeck.AsSlice()
	game := alphacats.NewRandomGame(deck, 4)
	result := countParallel(game, workCh)
	glog.Info(result)
}

type countJob struct {
	root     cfr.GameTreeNode
	resultCh chan int
	wg       *sync.WaitGroup
}

func doJob(job countJob, workCh chan countJob) {
	job.resultCh <- countParallel(job.root, workCh)
	job.wg.Done()
}

func countParallel(node cfr.GameTreeNode, workCh chan countJob) int {
	glog.Infof("Counting children for node: %v", node)
	node.(*alphacats.GameNode).Liberate()
	resultCh := make(chan int, node.NumChildren())
	var wg sync.WaitGroup
	for i := 0; i < node.NumChildren(); i++ {
		child := node.GetChild(i)
		child.(*alphacats.GameNode).Liberate()
		select {
		case workCh <- countJob{child, resultCh, &wg}:
			wg.Add(1)
		default:
			glog.Info("No workers available, counting children directly")
			workInProgress.Add(1)
			resultCh <- tree.CountNodes(child)
			workInProgress.Add(-1)
		}
	}

	go func() {
		glog.Info("Waiting for children to complete")
		wg.Wait()
		close(resultCh)
	}()

	// Do work as long as we are waiting for results.
	total := 0
	for {
		select {
		case job := <-workCh:
			doJob(job, workCh)
		case result, ok := <-resultCh:
			if !ok {
				glog.Infof("%d total children in subgame %v", total, node)
				return total
			}

			total += result
		}
	}
}
