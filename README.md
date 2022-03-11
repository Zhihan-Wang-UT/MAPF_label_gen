### Directory structure

```bash
dataGen # our repo
	build
	libtorch # need to download from https://pytorch.org/get-started/locally/ 
			 # select {stable | YourOS | LibTorch | C++/Java | CPU} as each options when download
	datagen.cpp
	CMakeLists.txt
	loadTorch.ipynb
datasets # from https://movingai.com/benchmarks/mapf/index.html 
	map
	scen_even
	scen_random
```

### Compile

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=$PWD/../libtorch ..
cmake --build .
```

### Run

```bash
cd build
./datagen ../../datasets/scen_random/empty-8-8-random-1.scen 10 out.pt
```