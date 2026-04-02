CXX = clang++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -Isrc

SRC = src/main.cc \
      src/builder/builder.cc \
	  src/ir/lower.cc \
	  src/codegen/lower.cc \
	  src/codegen/c_backend.cc

LIB_SRC = $(filter-out src/main.cc, $(SRC))

tenc: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

clean:
	rm -f tenc