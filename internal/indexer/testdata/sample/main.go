package main

import (
	"fmt"
	"strings"
)

type Saver interface {
	Save(s string) error
}

type FileRepo struct{}

func (r *FileRepo) Save(s string) error {
	fmt.Println(strings.ToUpper(s))
	return nil
}

type MemRepo struct{ items []string }

func (m MemRepo) Save(s string) error { return nil }

// Unused nunca é passado para process.
type Unused struct{}

func (Unused) Save(s string) error { return nil }

func process(s Saver, v string) error {
	defer cleanup()
	go metrics(v)
	return s.Save(v)
}

func cleanup() {}

func report(err error) string { return err.Error() }

func metrics(v string) {}

func Map[T, U any](in []T, f func(T) U) []U {
	out := make([]U, 0, len(in))
	for _, v := range in {
		out = append(out, f(v))
	}
	return out
}

func double(i int) int { return i * 2 }

func neverCalled() int { return 1 }

func main() {
	_ = process(&FileRepo{}, "a")
	_ = process(MemRepo{}, "b")
	_ = Map([]int{1}, double)
	run := func() { cleanup() }
	run()
	_ = report(fmt.Errorf("x"))
}
