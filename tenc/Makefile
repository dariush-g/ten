CXX = clang++
CXXFLAGS = -std=c++20 -O2 -Wall -Wextra -Isrc

SRC = src/main.cpp \
      src/builder/builder.cpp

tenc: $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $@

clean:
	rm -f tenc