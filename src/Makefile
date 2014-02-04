#G++FLAG = -g -Wall -Werror
G++FLAG = -O3 -m64 -mmmx -msse -msse2
I_PATH = /usr/local/include 
L_PATH = /usr/local/lib

LINK_OPTS = -L$(L_PATH) -lgsl -lgslcblas

all : gmm-testing

gmm-testing : gmm-testing.o
	g++ -o gmm-testing gmm-testing.o $(LINK_OPTS)

gmm-testing.o : gmm-testing.cpp
	g++ $(G++FLAG) -c gmm-testing.cpp -I$(I_PATH) 

clean :
	rm *.o
