# zuku: a futures library for Realm

zuku builds on top of the [Realm](https://gitlab.com/StanfordLegion/legion) event-based runtime system to provide a
futures-based API for expressing data-flow graphs. zuku provides two key
features on top of Realm:

* Capturing data effects on arrays and automatically building execution graphs based on
how data futures are used
* API for creating multi-GPU tasks (gang scheduling) and multi-GPU data movement (array resharding)

zuku provides a single-controller programming model.
The underlying implementation can either rely on control replication (MPI style)
to execute in a multi-controller fashion or execute with a true single controller
(although the single-controller is not yet implemented).

## Building

zuku uses a standard CMake build.
zuku must be pointed at a working Realm build or installation using the `-DLegion_ROOT` command.
A recommended build setup pointing to the Realm build tree would be:

```bash
legion=<Legion build directory>

cmake -S . -B build \
 -DLegion_ROOT=$legion \
 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
 -DCMAKE_CXX_STANDARD=17

cmake --build build
```

For configuring Realm, a recommended configure would be:

```bash
cmake \
  -S . \
  -B build \
  -GNinja \
  -D Legion_USE_CUDA=ON \
  -D Legion_BUILD_RUST_PROFILER=OFF \
  -D CMAKE_CXX_STANDARD=17 \
  -D CUDA_NVCC_FLAGS="-std=c++17" \
  -D CMAKE_EXPORT_COMPILE_COMMANDS=ON
```

With this configure, `<Legion build directory>` would be `<Legion source directory>/build`.

## High-level Overview

zuku provides a futures library on top of the Realm runtime providing deferred execution.
Rather than directly executing tasks, C++ lambdas are enqueued to run when their dependencies
are ready. The return values of the lambda are wrapped in futures and can be passed
to later tasks.

There are 3 main execution abstractions, which are largely inherited from Realm:
1. Processor
2. Memory
3. Ordering tokens

There are 3 main data abstractions:

1. `Future<T>`: an owning value holder for a deferred value
2. `View<T>`: a read-only value holder for a deferred value
3. `Store<T>`: a future with reference-like semantics for updating values in-place

Each of these objects is *move-only* and cannot be copied.
Views of each object, though, can be created by calling their `.view()` function,
which creates a new object holding read-only ownership of the underlying data.
An arbitrary number of views can be created.
All read-only views must be collected before a `Future` or `Store` is free
to modify the data. This is all managed automatically when
passing arguments to a `defer` function.

## Deferred Operations

A future is created by a deferred operation:

```c++
auto [f] = defer([]{
  return 42;
});
```
This `Future<int> f` can be passed to a further deferred task.
[See below](#tokens-structured-bindings) for the reasons behind the structured binding syntax.


```c++
auto token = defer([](int x){
  std::cerr << "Got " << x << std::endl;
}, std::move(f));
```

Here execution is ordered by implicit data dependencies.
The returned token can be used to wait on conditions
or provide execution dependencies.

This relinquishes ownership of `f` to the subtask.
If the task instead wants to read the data without
transferring ownership, a view can be created.

```c++
auto token = defer([](int x){
  std::cerr << "Got " << x << std::endl;
}, f.view());
```

To wait for all operations to finish, the user can do:
```c++
token.Wait();
```

### [Tokens and structured bindings](#tokens-structured-bindings)

The return type of each deferred operation is an ordering token
that can enforce execution dependencies. If the execution token
is not required, the user can unpack the token to the underlying
future objects using structured bindings.

## Stores

In contrast to `Future` which acts like `std::unique_ptr` and must transfer ownership
where used, `Store` provides something closer to `std::shared_ptr` or an
object passed by reference into a function. A `Store` can be modified in-place
by passing a child store to a deferred operation.

```c++
Store<int> a_st = Store<int>::Create();
defer([](int& a){
  a = 42;
}, a_st);
```
The task operates on a *reference*, which essentially provides an in-place update.
`Store` is move-only and cannot be copied. Instead, a child store is implicitly
created and passed to the deferred operation.

## Views

A `View` is a read-only handle for deferred data.
A `View` can be created from a `Future`, `Store`, or another `View`.

```c++
Store<int> a_st = Store<int>::Create();
defer([](const int& a){
  a = 42;
}, a_st);
```

In constrast to `Store`, the lambda can only take a const reference (or copy)
since the data is read-only.  The `defer` operation implicitly creates
a subview of the store and passes it to the deferred operation.

## Starting the Runtime

The simplest way to start the runtime and execute a program is to use the
`zuku::program` function.

```c++
int main(int argc, char** argv){
  return zuku::program([]{
    ...
  }, argc, argv);
}
```

`zuku::program` takes a lambda as an argument that executes the program.
A full "hello world" program using deferred execution would be:

```
int main(int argc, char** argv){
  return zuku::program([]{
    auto [msg] = defer([]{
      return "Hello World";
    });
    auto token = defer([](std::string name){
      std::cout << name << std::endl;
    }, std::move(msg));
    return token;
  }, argc, argv);
}
```

The `token` must be returned at the end to indicate the last
asynchronous event in the program. If `token` is not returned,
the program could end immediately without waiting for the subtasks.

## Bulk operations

zuku is designed to express bulk operations directly rather than having to express
parallelism via single-GPU operations.

### Device List

zuku provides a `DeviceList` that represents multiple processors:

```c++
DeviceList devices{{.start = 0, .num_devices= 8}};
```

This is the key abstraction for creating sharded arrays or gang-scheduling tasks.

### Sharded Array

Arrays are created with sharded shapes that defines how they should be arranged
across devices:

```c++
ShardedShape shape = {
  .type = zuku::SupportedType::S32,
  .sharding = {
    .dims = {
      { .size = 32, .sharding = 4, .permutation = 0},
      { .size = 64, .sharding = 2, .permutation = 1}
    },
    .devices = devices
  }
};
Store<ShardedArray> a = ShardedArray::Create(shape);
```

This creates an array with a 4x2 tiling across `devices`
with no permutation on the tile indexing.

### `across`

Tasks are gang-scheduled by specifying which devices to launch across:

```
across(devices).defer([](const ShardedArray& a){
  ...
}, a);
```
The input array sharding is expected to match the devices.
This launches a task on each participating device and passes
the individual shard to each task.

### `reshard`

Given two arrays `a` and `b` with different shardings,
the values in `a` can be copied to `b` using:

```c++
reshard(a,b)
```

Sharding encompasses both the tiling and the devices assigned.
A `reshard` can either involve a shape change, same shape on different devices,
or combination of the two.


## Etymology

*Zuku*nft: German for "future".

