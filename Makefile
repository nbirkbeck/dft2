all: dft dft_builder

dft_builder: dft_builder.c
	g++ -std=c++11 -g dft_builder.c -o dft_builder

dft: dft.c gen.h
	g++ -O3 -ffast-math -fabi-version=0 -mmmx -msse -mavx -mavx2  -std=c++11 dft.c  -lfftw3 -o dft

dft-clang: dft.c gen.h
	clang++ -O3 -ffast-math -mmmx -msse -mavx -mavx2  -std=c++11 dft.c  -lfftw3 -o dft-clang
